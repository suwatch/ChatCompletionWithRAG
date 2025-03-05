using System.Numerics.Tensors;
using System.Runtime.InteropServices.ComTypes;
using System.Text.Json;
using Microsoft.ML.Tokenizers;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Embeddings;
using Microsoft.SemanticKernel.Text;

namespace ChatCompletionWithRAG
{
    public class RAGFileInfo
    {
        public static readonly Lazy<Tokenizer> Tokenizer = new(() => TiktokenTokenizer.CreateForModel("text-embedding-3-small"));
        public static readonly string RAGCacheDirectory = Path.Combine(Environment.ExpandEnvironmentVariables(@"%USERPROFILE%\ChatCompletionWithRAGCache"));

        public string FullName { get; set; }
        public string CacheFile { get; set; }
        public int ChunkIndex { get; set; }
        public ReadOnlyMemory<float> Vector { get; set; }

        public float CompareVectors(ReadOnlyMemory<float> b)
        {
            var a = Vector;
            if (a.Length != b.Length || a.Span.Length == 0 || b.Span.Length == 0)
            {
                return -1;
            }

            return TensorPrimitives.CosineSimilarity(a.Span, b.Span);
            //return TensorPrimitives.Dot(a.Span, b.Span);
            //float result = 0;
            //for (int i = 0; i < a.Length; i++)
            //{
            //    result += a.Span[i] * b.Span[i];
            //}

            //return result;
        }

        public static async IAsyncEnumerable<RAGFileInfo> EnumerateDirectory(string directoryName, string searchPattern, Kernel kernel, ITextEmbeddingGenerationService textEmbeddingService, SearchOption option = SearchOption.AllDirectories)
        {
            var directory = new DirectoryInfo(directoryName);
            if (!directory.Exists)
            {
                throw new DirectoryNotFoundException($"Directory not found: {directoryName}");
            }

            var cacheDirectory = new DirectoryInfo(RAGCacheDirectory);
            if (!cacheDirectory.Exists)
            {
                cacheDirectory.Create();
            }

            SemaphoreSlim semaphore = new(4);
            var directoryNameHash = Fnv1aHashHelper.ComputeHash(directory.FullName.ToLowerInvariant());
            cacheDirectory = Directory.CreateDirectory(Path.Combine(cacheDirectory.FullName, $"{directoryNameHash:x8}"));
            foreach (var file in directory.EnumerateFiles(searchPattern, option))
            {
                var fileNameHash = Fnv1aHashHelper.ComputeHash(file.FullName.ToLowerInvariant());
                var fileDirectory = Directory.CreateDirectory(Path.Combine(cacheDirectory.FullName, $"{fileNameHash:x8}"));
                var cacheFiles = fileDirectory.GetFiles("*.json");
                var firstCacheFile = cacheFiles.FirstOrDefault();

                if (firstCacheFile is not null && firstCacheFile.LastWriteTimeUtc == file.LastWriteTimeUtc)
                {
                    Console.WriteLine($"Already learned for {file.FullName}");
                    int chunkIndex = 0;
                    foreach (var cacheFile in cacheFiles)
                    {
                        yield return new()
                        { 
                            FullName = file.FullName,
                            CacheFile = cacheFile.FullName,
                            ChunkIndex = chunkIndex,
                            Vector = JsonSerializer.Deserialize<ReadOnlyMemory<float>>(await File.ReadAllTextAsync(cacheFile.FullName).ConfigureAwait(false)) 
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
                            CacheFile = cacheFile.FullName,
                            ChunkIndex = chunkIndex,
                            Vector = vector
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
                var lines = TextChunker.SplitPlainTextLines(content, maxTokensPerLine: 10);
                var paragraphs = TextChunker.SplitPlainTextParagraphs(lines, maxTokensPerParagraph: 25);

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
    }
}