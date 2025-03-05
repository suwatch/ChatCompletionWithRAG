using System.ClientModel.Primitives;
using System.ComponentModel;
using System.Diagnostics;
using System.Net.Http.Headers;
using System.Text.Json;
using Azure.AI.OpenAI;
using Azure.Identity;
using Microsoft.Extensions.AI;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using Microsoft.SemanticKernel.Data;
using Microsoft.SemanticKernel.Embeddings;
using Microsoft.SemanticKernel.PromptTemplates.Handlebars;
using Newtonsoft.Json.Schema.Generation;

namespace ChatCompletionWithRAG
{
    internal static class Program
    {
        public const string DefaultFileExtensions = "*.txt,*.md";

        // set AZUREAI_NAME=your-azure-ai-name
        // make sure to Grant yourself "Cognitive Services OpenAI Contributor" RBAC to the AzureAI resource
        public static readonly string AzureAIName = EnsureEnvironmentVariable("AZUREAI_NAME");
        // set AZUREAI_DEPLOYMENTNAME=your-azure-ai-deployment-name
        public static readonly string AzureAIDeploymentName = EnsureEnvironmentVariable("AZUREAI_DEPLOYMENTNAME");
        // set AZUREAI_TEXTEMBEDDING_DEPLOYMENTNAME=your-azure-ai-textembedding-deployment-name
        public static readonly string AzureAITextEmbeddingDeploymentName = EnsureEnvironmentVariable("AZUREAI_TEXTEMBEDDING_DEPLOYMENTNAME");
        public static readonly TraceLevel HttpLoggingTraceLevel = int.TryParse(Environment.GetEnvironmentVariable("AZUREAI_HTTPLOGGING_TRACELEVEL"), out var value) ? (TraceLevel)value : TraceLevel.Off;

        static async Task Main(string[] args)
        {
            try
            {
                if (args.Length == 0)
                {
                    Console.WriteLine("Usage:");
                    Console.WriteLine("ChatCompletionWithRAG.exe directory");
                    Console.WriteLine("ChatCompletionWithRAG.exe directory *.txt,*.md");
                    Console.WriteLine("ChatCompletionWithRAG.exe CleanCache");
                    return;
                }
                else if (args[0] == "CleanCache")
                {
                    Directory.Delete(RAGFileInfo.RAGCacheDirectory, recursive: true);
                    Console.WriteLine($"{RAGFileInfo.RAGCacheDirectory} deleted");
                    return;
                }

                var directory = args[0];
                var patterns = args.Length > 1 ? args[1] : DefaultFileExtensions;

                var kernel = CreateKernelWithPlugins();
                var textEmbeddingService = kernel.GetRequiredService<ITextEmbeddingGenerationService>();

                var ragFiles = await LearnFromDirectoryAsync(directory, patterns, kernel, textEmbeddingService);

                kernel.Plugins.AddFromObject(new InformationProvider(ragFiles, kernel, textEmbeddingService), pluginName: "InformationProvider");

                //await RunChatLoop(kernel);

                await RunChatLoopWithTemplate(kernel);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex);
            }
        }

        static async Task RunChatLoop(Kernel kernel)
        {
            var chatClient = kernel.GetRequiredService<IChatCompletionService>();
            var executionSettings = new OpenAIPromptExecutionSettings { FunctionChoiceBehavior = FunctionChoiceBehavior.Auto() };
            while (true)
            {
                Console.Write("Question: ");
                var question = Console.ReadLine();
                if (string.IsNullOrWhiteSpace(question))
                {
                    break;
                }
                Console.WriteLine(question);

                var contents = await chatClient.GetChatMessageContentsAsync(question, executionSettings: executionSettings, kernel: kernel).WithProgress().ConfigureAwait(false);

                Console.Write("Answer: ");
                foreach (var content in contents)
                {
                    Console.WriteLine(content.Content);
                }

                Console.WriteLine();
            }
        }

        public class QuestionsResult
        {
            public string OriginalQuestion { get; set; }
            public string[] AlternativeQuestions { get; set; }
        }

        static async Task RunChatLoopWithTemplate(Kernel kernel)
        {
            var chatClient = kernel.GetRequiredService<IChatCompletionService>();
            var alternativeQuestionsTemplate = """
                You are an AI language model assistant. 
                Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database.
                By generating alternative questions, you can improve the quality of the retrieved documents.
                Original question: {{$originalQuestion}}
                """;

            // Specify response format by setting ChatResponseFormat object in prompt execution settings.
            var generator = new JSchemaGenerator();
            var jsonSchema = generator.Generate(typeof(QuestionsResult)).ToString();
            var executionSettings = new OpenAIPromptExecutionSettings
            {
                ResponseFormat = OpenAI.Chat.ChatResponseFormat.CreateJsonSchemaFormat(
                        "QuestionsResult",
                        BinaryData.FromString(jsonSchema))
            };

            var promptTemplate = """
                Answer the question based on the following information: 
                {{InformationProvider.GetInformation $query}}.
                Do include source references in your answer as footnotes.
                Question: {{$originalQuestion}}
                """;

            var promptSearchResultsTemplate = """
                Please use this information to answer the question:
                {{#with (InformationProvider-GetTextSearchResults query)}}  
                    {{#each this}}  
                        Name: {{Name}}
                        Value: {{Value}}
                        Link: {{Link}}
                        -----------------
                    {{/each}}
                {{/with}}

                Include link to the relevant source in the response as footnotes.
                    
                Question: {{originalQuestion}}
                """;

            while (true)
            {
                Console.Write("Question: ");
                var originalQuestion = "how long is MSI token cached?"; // Console.ReadLine();
                if (string.IsNullOrWhiteSpace(originalQuestion))
                {
                    break;
                }
                Console.WriteLine(originalQuestion);

                var result = await kernel.InvokePromptAsync(
                    promptTemplate: alternativeQuestionsTemplate,
                    arguments: new(executionSettings) { { "originalQuestion", originalQuestion } })
                    .WithProgress("QuestionTranslation").ConfigureAwait(false);

                var questionsResult = JsonSerializer.Deserialize<QuestionsResult>(result.ToString());
                for (int i = 0; i < questionsResult.AlternativeQuestions.Length; i++)
                {
                    Console.WriteLine($"Alternative Question[{i}]: {questionsResult.AlternativeQuestions[i]}");
                }

                //result = await kernel.InvokePromptAsync(
                //    promptTemplate: promptTemplate,
                //    arguments: new() { { "originalQuestion", originalQuestion }, { "query", result } })
                //    .WithProgress().ConfigureAwait(false);
                //Console.WriteLine($"Answer: {result}");

                var results = kernel.InvokePromptStreamingAsync(
                    promptTemplate: promptSearchResultsTemplate,
                    templateFormat: "handlebars",
                    promptTemplateFactory: new HandlebarsPromptTemplateFactory(),
                    arguments: new() { { "originalQuestion", originalQuestion }, { "query", questionsResult } });
                Console.Write($"Answer: ");
                await foreach (var res in results)
                {
                    Console.Write($"{res}");
                }

                Console.WriteLine();

                break;
            }
        }

        static async Task<List<RAGFileInfo>> LearnFromDirectoryAsync(string directory, string patterns, Kernel kernel, ITextEmbeddingGenerationService textEmbeddingService)
        {
            Console.WriteLine($"Learning from {directory} with patterns {patterns}");
            Console.WriteLine($"Caching at {RAGFileInfo.RAGCacheDirectory}");
            var results = new List<RAGFileInfo>();
            foreach (var pattern in patterns.Split([',', ';']))
            {
                await foreach (var ragFile in RAGFileInfo.EnumerateDirectory(directory, pattern, kernel, textEmbeddingService))
                {
                    results.Add(ragFile);
                }
            }
            return results;
        }

        private static Kernel CreateKernelWithPlugins(params Type[] pluginTypes)
        {
            var clientOptions = new AzureOpenAIClientOptions();
            if (HttpLoggingTraceLevel != TraceLevel.Off)
            {
                clientOptions.Transport = new HttpClientPipelineTransport(new HttpClient(new HttpLoggingHandler(verbose: HttpLoggingTraceLevel == TraceLevel.Verbose)));
            }

            var customClient = new AzureOpenAIClient(
                new Uri($"https://{AzureAIName}.openai.azure.com/"),
                DefaultAzureCredentialHelper.GetDefaultAzureCredential(AzureAuthorityHosts.AzurePublicCloud.AbsoluteUri),
                clientOptions);

            // Create a kernel with OpenAI chat completion and WeatherPlugin
            var builder = Kernel.CreateBuilder();
            builder.AddAzureOpenAIChatCompletion(AzureAIDeploymentName, customClient);
            builder.AddAzureOpenAITextEmbeddingGeneration(AzureAITextEmbeddingDeploymentName, azureOpenAIClient: customClient);

            foreach (var pluginType in pluginTypes)
            {
                builder.Plugins.Add(KernelPluginFactory.CreateFromType(pluginType));
            }

            return builder.Build();
        }

        private static string EnsureEnvironmentVariable(string name)
        {
            var value = Environment.GetEnvironmentVariable(name);
            if (string.IsNullOrWhiteSpace(value))
            {
                throw new InvalidOperationException($"Please set the {name} environment variable.");
            }
            return value;
        }

        internal static async Task<T> WithProgress<T>(this Task<T> task, string info = "", int dotIntercalSecs = 1)
        {
            Console.Write(info);
            while (task != await Task.WhenAny(Task.Delay(dotIntercalSecs * 1_000), task).ConfigureAwait(false))
            {
                Console.Write(".");
            }
            Console.WriteLine();

            return await task.ConfigureAwait(false);
        }

        private sealed class InformationProvider(IList<RAGFileInfo> ragFiles, Kernel kernel, ITextEmbeddingGenerationService textEmbeddingService)
        {
            [KernelFunction]
            [Description("Retrieve information and its reference to help answer any question.")]
            public async Task<string> GetInformation(
                [Description("The questions being asked")] QuestionsResult questionsResult)
            {
                var relevancies = new Dictionary<string, (int hits, float relevance, int chunkIndex)>(StringComparer.OrdinalIgnoreCase);
                var questionVectors = await textEmbeddingService.GenerateEmbeddingsAsync(questionsResult.AlternativeQuestions, kernel: kernel).ConfigureAwait(false);
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

                        Console.WriteLine($"[HIT#q{q}a{a}] Relevance: {result.relevance:0.00}, Total: {relevancies[result.file.FullName]:0.00}, file: {result.file.FullName}[chunk#{result.file.ChunkIndex}]");
                    }
                }

                var mostRelevance = relevancies.OrderByDescending(r => r.Value).First();

                Console.WriteLine($"[BestMatchedDoc] Hits: {mostRelevance.Value.hits} out of {2 * questionVectors.Count}, Relevance: {mostRelevance.Value.relevance:0.00}, file: {mostRelevance.Key}[chunk#{mostRelevance.Value.chunkIndex}]");

                return JsonSerializer.Serialize(new { AssistantMessage = File.ReadAllText(mostRelevance.Key), Reference = mostRelevance.Key });
            }

            [KernelFunction]
            [Description("Retrieve top search results to help answer any question.")]
            public async Task<IEnumerable<TextSearchResult>> GetTextSearchResults(
                [Description("The questions being asked")] QuestionsResult questionsResult)
            {
                var relevancies = new Dictionary<string, (int hits, float relevance, int chunkIndex)>(StringComparer.OrdinalIgnoreCase);
                var questionVectors = await textEmbeddingService.GenerateEmbeddingsAsync(questionsResult.AlternativeQuestions, kernel: kernel).ConfigureAwait(false);
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

                        Console.WriteLine($"[HIT#q{q}a{a}] Relevance: {result.relevance:0.00}, Total: {relevancies[result.file.FullName]:0.00}, file: {result.file.FullName}[chunk#{result.file.ChunkIndex}]");
                    }
                }

                var searchResults = new List<TextSearchResult>();
                foreach (var relevance in relevancies.OrderByDescending(r => r.Value).Take(1))
                {
                    Console.WriteLine($"[BestMatchedDoc] Hits: {relevance.Value.hits} out of {2 * questionVectors.Count}, Relevance: {relevance.Value.relevance:0.00}, file: {relevance.Key}[chunk#{relevance.Value.chunkIndex}]");
                    var content = await File.ReadAllTextAsync(relevance.Key).ConfigureAwait(false);
                    searchResults.Add(new TextSearchResult(content) { Link = relevance.Key, Name = relevance.Key });
                }

                return searchResults;
            }
        }
    }
}