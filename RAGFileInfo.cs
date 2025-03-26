using System.Linq.Expressions;
using System.Numerics.Tensors;
using System.Text;
using System.Text.Json;
using Azure.Identity;
using Azure.Storage.Blobs;
using Azure.Storage.Blobs.Models;
using Microsoft.Extensions.VectorData;
using Microsoft.ML.Tokenizers;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Embeddings;
using Microsoft.SemanticKernel.Text;
using SemanticSlicer;
using SemanticSlicer.Models;
using static Microsoft.SemanticKernel.Text.TextChunker;

namespace ChatCompletionWithRAG
{
    public class RAGFileInfo
    {
        public static readonly Lazy<Tokenizer> Tokenizer = new(() => TiktokenTokenizer.CreateForModel("text-embedding-3-small"));

        private string _key;
        private string _fileHash;

        [VectorStoreRecordKey]
        public string Key { get => (_key ?? (_key = $"{FileHash}-{ChunkIndex:0000}")); set => _key = value; }
        [VectorStoreRecordData(IsFilterable = true)]
        public string FullName { get; set; }
        [VectorStoreRecordData]
        public int ChunkIndex { get; set; }
        [VectorStoreRecordData]
        public DateTimeOffset? LastModifiedTimeUtc { get; set; }
        [VectorStoreRecordVector(1536)]
        public ReadOnlyMemory<float> TextEmbedding { get; set; }

        public string FileHash => _fileHash ?? (_fileHash = $"{Fnv1aHashHelper.ComputeHash(FullName.ToLowerInvariant()):x8}");

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

            cacheDirectory = Directory.CreateDirectory(Path.Combine(cacheDirectory.FullName, collection.CollectionName));
            foreach (var file in directory.EnumerateFiles(searchPattern, option))
            {
                var fileHash = $"{Fnv1aHashHelper.ComputeHash(file.FullName.ToLowerInvariant()):x8}";
                var result = await collection.GetAsync($"{fileHash}-0000").ConfigureAwait(false);
                if (result is not null)
                {
                    if (result.LastModifiedTimeUtc == file.LastWriteTimeUtc)
                    {
                        Program.WriteLine($"Already exists in VectorStore {file.FullName}");
                        continue;
                    }

                    Program.WriteLine($"[WARNING] Mismatch LastWriteTimeUtc in VectorStore {file.FullName}");
                    continue;
                }

                var fileDirectory = Directory.CreateDirectory(Path.Combine(cacheDirectory.FullName, fileHash));
                var cacheFiles = fileDirectory.GetFiles("*.json");
                var firstCacheFile = cacheFiles.FirstOrDefault();

                if (firstCacheFile is not null && firstCacheFile.LastWriteTimeUtc == file.LastWriteTimeUtc)
                {
                    Program.WriteLine($"Already learned {file.FullName}");
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
                    Program.WriteLine($"New learning for {file.FullName}");
                    if (firstCacheFile is not null)
                    {
                        fileDirectory.Delete(recursive: true);
                        fileDirectory = Directory.CreateDirectory(Path.Combine(cacheDirectory.FullName, fileHash));
                    }

                    int chunkIndex = 0;
                    await foreach (var vector in RAGFileInfo.FromFileAsync(file.FullName, kernel, textEmbeddingService).ConfigureAwait(false))
                    {
                        if (vector.Length == 0)
                            continue;

                        var cacheFile = new FileInfo(Path.Combine(fileDirectory.FullName, $"{chunkIndex:0000}.json"));
                        var ragInfo = new RAGFileInfo()
                        {
                            FullName = file.FullName,
                            ChunkIndex = chunkIndex,
                            LastModifiedTimeUtc = file.LastWriteTimeUtc,
                            TextEmbedding = vector,
                        };

                        if (chunkIndex == 0)
                        {
                            await ragInfo.UploadBlobAsync(collection.CollectionName);
                        }
                        await File.WriteAllTextAsync(cacheFile.FullName, JsonSerializer.Serialize(vector)).ConfigureAwait(false);
                        File.SetLastWriteTimeUtc(cacheFile.FullName, file.LastWriteTimeUtc);

                        yield return ragInfo;
                        chunkIndex++;
                    }
                }
            }
        }

        public static async IAsyncEnumerable<ReadOnlyMemory<float>> FromFileAsync(string file, Kernel kernel, ITextEmbeddingGenerationService textEmbeddingService)
        {
            // To demonstrate batching we'll create abnormally small partitions.
            var content = await File.ReadAllTextAsync(file).ConfigureAwait(false);
            if (string.IsNullOrWhiteSpace(content))
            {
                yield return new ReadOnlyMemory<float>();
            }
            else
            {
                List<DocumentChunk> documentChunks;
                var useTextChunker = true;
                if (useTextChunker)
                {
                    TokenCounter tokenCounter = new(c => Tokenizer.Value.CountTokens(c));
                    var lines = file.EndsWith(".md", StringComparison.OrdinalIgnoreCase)
                        ? TextChunker.SplitMarkDownLines(content, maxTokensPerLine: 50, tokenCounter: tokenCounter)
                        : TextChunker.SplitPlainTextLines(content, maxTokensPerLine: 50, tokenCounter: tokenCounter);
                    var paragraphs = file.EndsWith(".md", StringComparison.OrdinalIgnoreCase)
                        ? TextChunker.SplitMarkdownParagraphs(lines, maxTokensPerParagraph: 100, tokenCounter: tokenCounter)
                        : TextChunker.SplitPlainTextParagraphs(lines, maxTokensPerParagraph: 100, tokenCounter: tokenCounter);

                    documentChunks = paragraphs.Select(p => new DocumentChunk
                    {
                        Content = p,
                        TokenCount = tokenCounter(p),
                    }).ToList();
                }
                else
                {
                    var options = new SlicerOptions { MaxChunkTokenCount = 100, StripHtml = true, MinChunkPercentage = 30 };
                    if (file.EndsWith(".md", StringComparison.OrdinalIgnoreCase))
                    {
                        options.Separators = Separators.Markdown;
                    }
                    else if (file.EndsWith(".html", StringComparison.OrdinalIgnoreCase)
                        || file.EndsWith(".htm", StringComparison.OrdinalIgnoreCase))
                    {
                        options.Separators = Separators.Html;
                    }

                    var slicer = new Slicer(options);
                    try
                    {
                        documentChunks = slicer.GetDocumentChunks(content);
                    }
                    catch (Exception ex)
                    {
                        Program.WriteLine($"SemanticSlicer for {file}  {ex}");
                        documentChunks = new();
                    }
                }

                // Azure OpenAI currently supports input arrays up to 16 for text-embedding-ada-002 (Version 2).
                // Both require the max input token limit per API request to remain under 8191 for this model.
                var chunks = documentChunks
                    .ChunkByAggregate(
                        seed: 0,
                        aggregator: (tokenCount, paragraph) => tokenCount + paragraph.TokenCount,
                        predicate: (tokenCount, index) => tokenCount < 8191 && index < 16)
                    .ToList();

                Program.WriteLine($"SemanticSlicer for {file}, chunks={chunks.Count}, slices={documentChunks.Count}, tokenCounts='{string.Join(",", documentChunks.Select(c => c.TokenCount))}'");

                var semaphore = new SemaphoreSlim(4);
                var tasks = new List<Task<IList<ReadOnlyMemory<float>>>>(chunks.Count);
                for (var i = 0; i < chunks.Count; ++i)
                {
                    tasks.Add(SafeGenerateEmbeddingsAsync($"{file}#{i:0000}", chunks[i].Select(c => c.Content).ToList(), kernel: kernel, textEmbeddingService, semaphore));
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
                Program.WriteLine($"Error {ex} generating embedding for {chunkInfo}");
                throw;
            }
            finally
            {
                semaphore.Release();
            }
        }

        public Stream GetStream()
            => new FileInfo(FullName).OpenRead();

        public async Task UploadBlobAsync(string containerName)
        {
            var blobServiceClient = new BlobServiceClient(new Uri($"https://{Program.AzureAIStorageName}.blob.core.windows.net/"), DefaultAzureCredentialHelper.GetDefaultAzureCredential(AzureAuthorityHosts.AzurePublicCloud.AbsoluteUri));
            var containerClient = blobServiceClient.GetBlobContainerClient(containerName);
            var blobName = FileHash;
            var blobClient = containerClient.GetBlobClient(blobName);
            var lastModifiedTimeUtcString = $"{LastModifiedTimeUtc:s}";
            try
            {
                var properties = await blobClient.GetPropertiesAsync();
                if (properties.Value.Metadata.TryGetValue("SourceLastModifiedTimeUtc", out var value)
                    && lastModifiedTimeUtcString == value)
                {
                    Program.WriteLine($"Already uploaded {containerName}/{blobName} from {FullName}");
                    return;
                }
            }
            catch (Azure.RequestFailedException ex) when (ex.Status == 404 && ex.ErrorCode == "ContainerNotFound")
            {
                await containerClient.CreateIfNotExistsAsync();
            }
            catch (Azure.RequestFailedException ex) when (ex.Status == 404 && ex.ErrorCode == "BlobNotFound")
            {
                // no-op
            }

            using var stream = GetStream();
            await blobClient.UploadAsync(stream, overwrite: true);
            await blobClient.SetMetadataAsync(new Dictionary<string, string>
            {
                { "SourceName", FullName },
                { "SourceLastModifiedTimeUtc", lastModifiedTimeUtcString },
            });

            Program.WriteLine($"Successfully uploaded {containerName}/{blobName} to {FullName}");
        }

        public async Task<string> DownloadBlobAsync(string containerName)
        {
            var blobServiceClient = new BlobServiceClient(new Uri($"https://{Program.AzureAIStorageName}.blob.core.windows.net/"), DefaultAzureCredentialHelper.GetDefaultAzureCredential(AzureAuthorityHosts.AzurePublicCloud.AbsoluteUri));
            var containerClient = blobServiceClient.GetBlobContainerClient(containerName);
            var blobName = FileHash;
            var blobClient = containerClient.GetBlobClient(blobName);
            var result = await blobClient.DownloadContentAsync();
            return result.Value.Content.ToString();
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