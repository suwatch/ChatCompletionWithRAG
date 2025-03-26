using System.ComponentModel;
using System.Text.Json;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Embeddings;

namespace ChatCompletionWithRAG
{
    public sealed class LocalFileSearchProvider(IList<RAGFileInfo> ragFiles, ITextEmbeddingGenerationService textEmbeddingService)
    {
        [KernelFunction]
        [Description("Retrieve information and its reference to help answer any question.")]
        public async Task<string> GetInformation(
            [Description("The questions being asked")] QuestionsResult questionsResult)
        {
            var relevancies = new Dictionary<string, (int hits, float relevance, int chunkIndex)>(StringComparer.OrdinalIgnoreCase);
            var questionVectors = await textEmbeddingService.GenerateEmbeddingsAsync(questionsResult.AlternativeQuestions).ConfigureAwait(false);
            //var questionVectors = await textEmbeddingService.GenerateEmbeddingsAsync(questionsResult.AlternativeQuestions, kernel: kernel).ConfigureAwait(false);
            for (int q = 0; q < questionVectors.Count; ++q)
            {
                var questionVector = questionVectors[q];
                var question = questionsResult.AlternativeQuestions[q];
                var results = ragFiles
                    .Select(f => (file: f, relevance: f.CompareVectors(questionVector), question: question))
                    .OrderByDescending(f => f.relevance)
                    .Take(2).ToList();
                for (int a = 0; a < results.Count; ++a)
                {
                    var result = results[a];
                    relevancies[result.file.FullName] = relevancies.TryGetValue(result.file.FullName, out var value)
                        ? (value.hits + 1, value.relevance + result.relevance, value.chunkIndex)
                        : (1, result.relevance, result.file.ChunkIndex);

                    Program.WriteLine($"[HIT#q{q}a{a}] Relevance: {result.relevance:0.00}, Total: {relevancies[result.file.FullName]:0.00}, file: {result.file.FullName}#{result.file.ChunkIndex:0000}]");
                }
            }

            var mostRelevance = relevancies.OrderByDescending(r => r.Value).First();

            Program.WriteLine($"[BestMatchedDoc] Hits: {mostRelevance.Value.hits} out of {2 * questionVectors.Count}, Relevance: {mostRelevance.Value.relevance:0.00}, file: {mostRelevance.Key}#{mostRelevance.Value.chunkIndex:0000}");

            return JsonSerializer.Serialize(new { AssistantMessage = File.ReadAllText(mostRelevance.Key), Reference = mostRelevance.Key });
        }
    }
}
