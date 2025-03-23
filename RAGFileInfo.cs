using System.Numerics.Tensors;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.VectorData;
using Microsoft.ML.Tokenizers;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Embeddings;
using Microsoft.SemanticKernel.Text;
using static Microsoft.SemanticKernel.Text.TextChunker;

namespace ChatCompletionWithRAG
{
    public class RAGFileInfo
    {
        public static readonly Lazy<Tokenizer> Tokenizer = new(() => TiktokenTokenizer.CreateForModel("text-embedding-3-small"));

        private string _key;

        [VectorStoreRecordKey]
        //public uint Key { get => (_key ?? Fnv1aHashHelper.ComputeHash(Encoding.UTF8.GetBytes($"{FullName}#{ChunkIndex}"))); set => _key = value; }
        public string Key { get => (_key ?? (_key  = $"{Fnv1aHashHelper.ComputeHash(FullName.ToLowerInvariant()):x8}-{ChunkIndex:0000}")); set => _key = value; }
        [VectorStoreRecordData(IsFilterable = true)]
        public string FullName { get; set; }
        [VectorStoreRecordData]
        public int ChunkIndex { get; set; }
        [VectorStoreRecordData]
        public DateTimeOffset? LastModifiedTimeUtc { get; set; }
        [VectorStoreRecordVector(1536)]
        public ReadOnlyMemory<float> TextEmbedding { get; set; }

        public float CompareVectors(ReadOnlyMemory<float> b)
        {
            var a = TextEmbedding;
            if (a.Length != b.Length || a.Span.Length == 0 || b.Span.Length == 0)
            {
                return -1;
            }

            return TensorPrimitives.CosineSimilarity(a.Span, b.Span);
        }

        public static async IAsyncEnumerable<RAGFileInfo> EnumerateDirectory(IVectorStoreRecordCollection<string, RAGFileInfo> collection, string directoryName, string searchPattern, Kernel kernel, ITextEmbeddingGenerationService textEmbeddingService, SearchOption option = SearchOption.AllDirectories)
        {
            var directory = new DirectoryInfo(directoryName);
            if (!directory.Exists)
            {
                throw new DirectoryNotFoundException($"Directory not found: {directoryName}");
            }

            var cacheDirectory = new DirectoryInfo(Program.RAGCacheDirectory);
            if (!cacheDirectory.Exists)
            {
                cacheDirectory.Create();
            }

            var directoryNameHash = Fnv1aHashHelper.ComputeHash(directory.FullName.ToLowerInvariant());
            cacheDirectory = Directory.CreateDirectory(Path.Combine(cacheDirectory.FullName, $"{directoryNameHash:x8}"));
            foreach (var file in directory.EnumerateFiles(searchPattern, option))
            {
                var fileNameHash = Fnv1aHashHelper.ComputeHash(file.FullName.ToLowerInvariant());
                var result = await collection.GetAsync($"{fileNameHash:x8}-0000").ConfigureAwait(false);
                if (result is not null)
                {
                    if (result.LastModifiedTimeUtc == file.LastWriteTimeUtc)
                    {
                        Console.WriteLine($"Already exists in VectorStore {file.FullName}");
                        continue;
                    }

                    Console.WriteLine($"[WARNING] Mismatch LastWriteTimeUtc in VectorStore {file.FullName}");
                    continue;
                }

                var fileDirectory = Directory.CreateDirectory(Path.Combine(cacheDirectory.FullName, $"{fileNameHash:x8}"));
                var cacheFiles = fileDirectory.GetFiles("*.json");
                var firstCacheFile = cacheFiles.FirstOrDefault();

                if (firstCacheFile is not null && firstCacheFile.LastWriteTimeUtc == file.LastWriteTimeUtc)
                {
                    Console.WriteLine($"Already learned {file.FullName}");
                    int chunkIndex = 0;
                    foreach (var cacheFile in cacheFiles)
                    {
                        yield return new()
                        { 
                            FullName = file.FullName,
                            ChunkIndex = chunkIndex,
                            LastModifiedTimeUtc = file.LastWriteTimeUtc,
                            TextEmbedding = JsonSerializer.Deserialize<ReadOnlyMemory<float>>(await File.ReadAllTextAsync(cacheFile.FullName).ConfigureAwait(false)),
                        };

                        chunkIndex++;
                    }
                }
                else
                {
                    Console.WriteLine($"New learning for {file.FullName}");
                    if (firstCacheFile is not null)
                    {
                        fileDirectory.Delete(recursive: true);
                        fileDirectory = Directory.CreateDirectory(Path.Combine(cacheDirectory.FullName, $"{fileNameHash:x8}"));
                    }

                    int chunkIndex = 0;
                    await foreach (var vector in RAGFileInfo.FromFileAsync(file.FullName, kernel, textEmbeddingService).ConfigureAwait(false))
                    {
                        var cacheFile = new FileInfo(Path.Combine(fileDirectory.FullName, $"{chunkIndex:0000}.json"));
                        await File.WriteAllTextAsync(cacheFile.FullName, JsonSerializer.Serialize(vector)).ConfigureAwait(false);
                        File.SetLastWriteTimeUtc(cacheFile.FullName, file.LastWriteTimeUtc);
                        yield return new()
                        {
                            FullName = file.FullName,
                            ChunkIndex = chunkIndex,
                            LastModifiedTimeUtc = file.LastWriteTimeUtc,
                            TextEmbedding = vector,
                        };

                        chunkIndex++;
                    }
                }
            }
        }

        public static async IAsyncEnumerable<ReadOnlyMemory<float>> FromFileAsync(string file, Kernel kernel, ITextEmbeddingGenerationService textEmbeddingService)
        {
            var tokenizer = Tokenizer.Value;

            // To demonstrate batching we'll create abnormally small partitions.
            var content = await File.ReadAllTextAsync(file).ConfigureAwait(false);
            if (string.IsNullOrWhiteSpace(content))
            {
                yield return new ReadOnlyMemory<float>();
            }
            else
            {
                TokenCounter tokenCounter = new(c => tokenizer.CountTokens(c));
                var lines = file.EndsWith(".md", StringComparison.OrdinalIgnoreCase)
                    ? TextChunker.SplitMarkDownLines(content, maxTokensPerLine: 15, tokenCounter: tokenCounter)
                    : TextChunker.SplitPlainTextLines(content, maxTokensPerLine: 15, tokenCounter: tokenCounter);
                var paragraphs = file.EndsWith(".md", StringComparison.OrdinalIgnoreCase)
                    ? TextChunker.SplitMarkdownParagraphs(lines, maxTokensPerParagraph: 40, tokenCounter: tokenCounter)
                    : TextChunker.SplitPlainTextParagraphs(lines, maxTokensPerParagraph: 40, tokenCounter: tokenCounter);

                // Azure OpenAI currently supports input arrays up to 16 for text-embedding-ada-002 (Version 2).
                // Both require the max input token limit per API request to remain under 8191 for this model.
                var chunks = paragraphs
                    .ChunkByAggregate(
                        seed: 0,
                        aggregator: (tokenCount, paragraph) => tokenCount + tokenizer.CountTokens(paragraph),
                        predicate: (tokenCount, index) => tokenCount < 8191 && index < 16)
                    .ToList();

                var semaphore = new SemaphoreSlim(4);
                var tasks = new List<Task<IList<ReadOnlyMemory<float>>>>(chunks.Count);
                for (var i = 0; i < chunks.Count; ++i)
                {
                    tasks.Add(SafeGenerateEmbeddingsAsync($"{file}[chunk#{i}]", chunks[i], kernel: kernel, textEmbeddingService, semaphore));
                }

                await Task.WhenAll(tasks).ConfigureAwait(false);

                foreach (var task in tasks.Where(t => t.IsCompletedSuccessfully))
                {
                    foreach (var embedding in task.Result)
                    {
                        yield return embedding;
                    }
                }
            }
        }

        private static async Task<IList<ReadOnlyMemory<float>>> SafeGenerateEmbeddingsAsync(string chunkInfo, IList<string> content, Kernel kernel, ITextEmbeddingGenerationService textEmbeddingService, SemaphoreSlim semaphore)
        {
            await semaphore.WaitAsync().ConfigureAwait(false);
            try
            {
                return await textEmbeddingService.GenerateEmbeddingsAsync(content, kernel: kernel).WithProgress(chunkInfo).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error {ex} generating embedding for {chunkInfo}");
                return [new ReadOnlyMemory<float>()];
            }
            finally
            {
                semaphore.Release();
            }
        }

        public sealed class StatefulTokenCounter
        {
            private readonly Dictionary<string, int> _callStats = [];
            private readonly Tokenizer _tokenizer = TiktokenTokenizer.CreateForModel("gpt-4");

            public int Count(string input)
            {
                this.CallCount++;
                this._callStats[input] = this._callStats.TryGetValue(input, out int value) ? value + 1 : 1;
                return this._tokenizer.CountTokens(input);
            }

            public int CallCount { get; private set; } = 0;
        }

        private static TokenCounter StatelessTokenCounter => (string input) =>
        {
            var tokenizer = TiktokenTokenizer.CreateForModel("gpt-4");
            return tokenizer.CountTokens(input);
        };
    }
}