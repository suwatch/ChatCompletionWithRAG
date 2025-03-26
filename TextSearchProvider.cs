using System.ComponentModel;
using Microsoft.Extensions.VectorData;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Data;
using Microsoft.SemanticKernel.Embeddings;

namespace ChatCompletionWithRAG
{
    public sealed class TextSearchProvider(IVectorStoreRecordCollection<string, RAGFileInfo> collection, ITextEmbeddingGenerationService textEmbeddingService)
    {
        [KernelFunction]
        [Description("Retrieve top search results to help answer any question.")]
        public async Task<IEnumerable<TextSearchResult>> GetTextSearchResults(
            [Description("The questions being asked")] QuestionsResult questionsResult)
        {
            var relevancies = new Dictionary<string, (int hits, double relevance, RAGFileInfo info)>(StringComparer.OrdinalIgnoreCase);
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
                        relevancies[result.Key] = relevancies.TryGetValue(result.Key, out var value)
                            ? (value.hits + 1, value.relevance + vectorResult.Score.Value, value.info)
                            : (1, vectorResult.Score.Value, result);

                        Program.WriteLine($"[HIT#q{q}a{a}] Relevance: {vectorResult.Score:0.00}, Total: {relevancies[result.Key].relevance:0.00}, file: {result.FullName}#{result.ChunkIndex:0000}");
                        ++a;
                    }
                }
            }
            else
            {
                var questionVectors = await textEmbeddingService.GenerateEmbeddingsAsync(questionsResult.AlternativeQuestions).ConfigureAwait(false);
                //var questionVectors = await textEmbeddingService.GenerateEmbeddingsAsync(questionsResult.AlternativeQuestions, kernel: kernel).ConfigureAwait(false);
                var tasks = new List<Task<(string question, List<VectorSearchResult<RAGFileInfo>> results)>>();
                for (int q = 0; q < questionVectors.Count; ++q)
                {
                    tasks.Add(VectorizedSearchAsync(questionsResult.AlternativeQuestions[q], questionVectors[q]));
                }
                await Task.WhenAll(tasks);

                foreach (var taskResult in tasks.Select(t => t.Result))
                {
                    var question = taskResult.question;
                    foreach (var vectorResult in taskResult.results)
                    {
                        relevancies[vectorResult.Record.Key] = relevancies.TryGetValue(vectorResult.Record.Key, out var value)
                            ? (value.hits + 1, value.relevance + vectorResult.Score.Value, value.info)
                            : (1, vectorResult.Score.Value, vectorResult.Record);

                        Program.WriteLine($"[HIT {question}] Relevance: {vectorResult.Score:0.00}, Total: {relevancies[vectorResult.Record.Key].relevance:0.00}, file: {vectorResult.Record.FullName}#{vectorResult.Record.ChunkIndex:0000}");
                    }
                }
            }

            var searchResults = new List<TextSearchResult>();
            foreach (var relevance in relevancies.OrderByDescending(r => r.Value.relevance).Take(1))
            {
                Program.WriteLine($"[BestMatchedDoc] Hits: {relevance.Value.hits} out of {2 * questionsResult.AlternativeQuestions.Length}, Relevance: {relevance.Value.relevance:0.00}, file: {relevance.Value.info.FullName}#{relevance.Value.info.ChunkIndex:0000}");
                var content = await relevance.Value.info.DownloadBlobAsync(containerName: collection.CollectionName);
                searchResults.Add(new TextSearchResult(content) { Link = relevance.Value.info.FullName, Name = relevance.Value.info.FullName });
            }

            return searchResults;
        }

        private async Task<(string question, List<VectorSearchResult<RAGFileInfo>> results)> VectorizedSearchAsync(string question, ReadOnlyMemory<float> questionVector)
        {
            List<RAGFileInfo> results = new List<RAGFileInfo>(2);
            var vectorResults = await collection.VectorizedSearchAsync(questionVector, new() { Top = 2 });
            return (question, await vectorResults.Results.ToListAsync());
        }
    }
}
