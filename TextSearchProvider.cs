using System.ComponentModel;
using Microsoft.Extensions.VectorData;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Data;
using Microsoft.SemanticKernel.Embeddings;

namespace ChatCompletionWithRAG
{
    public sealed class TextSearchProvider(IVectorStoreRecordCollection<string, RAGFileInfo> collection, Kernel kernel, ITextEmbeddingGenerationService textEmbeddingService)
    {
        [KernelFunction]
        [Description("Retrieve top search results to help answer any question.")]
        public async Task<IEnumerable<TextSearchResult>> GetTextSearchResults(
            [Description("The questions being asked")] QuestionsResult questionsResult)
        {
            var relevancies = new Dictionary<string, (int hits, double relevance, int chunkIndex)>(StringComparer.OrdinalIgnoreCase);
            if (collection is IVectorizableTextSearch<RAGFileInfo> search)
            {
                for (int q = 0; q < questionsResult.AlternativeQuestions.Length; ++q)
                {
                    var question = questionsResult.AlternativeQuestions[q];
                    var vectorResults = await search.VectorizableTextSearchAsync(question, new() { Top = 2 });
                    int a = 0;
                    await foreach (var vectorResult in vectorResults.Results)
                    {
                        var result = vectorResult.Record;
                        relevancies[result.FullName] = relevancies.TryGetValue(result.FullName, out var value)
                            ? (value.hits + 1, value.relevance + vectorResult.Score.Value, value.chunkIndex)
                            : (1, vectorResult.Score.Value, result.ChunkIndex);

                        Console.WriteLine($"[HIT#q{q}a{a}] Relevance: {vectorResult.Score:0.00}, Total: {relevancies[result.FullName].relevance:0.00}, file: {result.FullName}[chunk#{result.ChunkIndex}]");
                        ++a;
                    }
                }
            }
            else
            {
                var questionVectors = await textEmbeddingService.GenerateEmbeddingsAsync(questionsResult.AlternativeQuestions, kernel: kernel).ConfigureAwait(false);
                for (int q = 0; q < questionVectors.Count; ++q)
                {
                    var questionVector = questionVectors[q];
                    var question = questionsResult.AlternativeQuestions[q];

                    var vectorResults = await collection.VectorizedSearchAsync(questionVector, new() { Top = 2 });
                    int a = 0;
                    await foreach (var vectorResult in vectorResults.Results)
                    {
                        var result = vectorResult.Record;
                        relevancies[result.FullName] = relevancies.TryGetValue(result.FullName, out var value)
                            ? (value.hits + 1, value.relevance + vectorResult.Score.Value, value.chunkIndex)
                            : (1, vectorResult.Score.Value, result.ChunkIndex);

                        Console.WriteLine($"[HIT#q{q}a{a}] Relevance: {vectorResult.Score:0.00}, Total: {relevancies[result.FullName].relevance:0.00}, file: {result.FullName}[chunk#{result.ChunkIndex}]");
                        ++a;
                    }
                }
            }

            var searchResults = new List<TextSearchResult>();
            foreach (var relevance in relevancies.OrderByDescending(r => r.Value).Take(1))
            {
                Console.WriteLine($"[BestMatchedDoc] Hits: {relevance.Value.hits} out of {2 * questionsResult.AlternativeQuestions.Length}, Relevance: {relevance.Value.relevance:0.00}, file: {relevance.Key}[chunk#{relevance.Value.chunkIndex}]");
                var content = await File.ReadAllTextAsync(relevance.Key).ConfigureAwait(false);
                searchResults.Add(new TextSearchResult(content) { Link = relevance.Key, Name = relevance.Key });
            }

            return searchResults;
        }
    }
}
