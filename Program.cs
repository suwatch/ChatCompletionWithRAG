using System.ClientModel.Primitives;
using System.ComponentModel;
using System.Diagnostics;
using System.Text;
using System.Text.Json;
using Azure.AI.OpenAI;
using Azure.Core;
using Azure.Identity;
using Microsoft.Azure.Cosmos;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.VectorData;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.InMemory;
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
        public const string VectorDatabaseName = "antragdb";

        // set AZUREAI_NAME=your-azure-ai-name
        // make sure to Grant yourself "Cognitive Services OpenAI Contributor" RBAC to the AzureAI resource
        public static readonly string AzureAIName = EnsureEnvironmentVariable("AZUREAI_NAME");
        // set AZUREAI_DEPLOYMENTNAME=your-azure-ai-deployment-name
        public static readonly string AzureAIDeploymentName = EnsureEnvironmentVariable("AZUREAI_DEPLOYMENTNAME");
        // set AZUREAI_TEXTEMBEDDING_DEPLOYMENTNAME=your-azure-ai-textembedding-deployment-name
        public static readonly string AzureAITextEmbeddingDeploymentName = EnsureEnvironmentVariable("AZUREAI_TEXTEMBEDDING_DEPLOYMENTNAME");
        public static readonly TraceLevel HttpLoggingTraceLevel = Enum.TryParse<TraceLevel>(Environment.GetEnvironmentVariable("AZUREAI_HTTPLOGGING_TRACELEVEL"), out var value) ? value : TraceLevel.Off;
        public static readonly TraceLevel DebugTraceLevel = Enum.TryParse<TraceLevel>(Environment.GetEnvironmentVariable("AZUREAI_TRACELEVEL"), out var value) ? value : TraceLevel.Off;
        public static readonly string RAGCacheDirectory = !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("AZUREAI_CACHE_DIRECTORY"))
            ? Environment.GetEnvironmentVariable("AZUREAI_CACHE_DIRECTORY") : Path.Combine(Environment.ExpandEnvironmentVariables(@"%USERPROFILE%\ChatCompletionWithRAGCache"));

        static Dictionary<string, Func<string[], Task>> _methods = new(StringComparer.OrdinalIgnoreCase)
        {
            { nameof(Usage), Usage },
            { nameof(Learn), Learn },
            { nameof(Assist), Assist },
            { nameof(CleanCache), CleanCache },
        };

        static async Task Main(string[] args)
        {
            try
            {
                //var cosmosClient = new CosmosClient(
                //    accountEndpoint: "https://kuduaicosmosdbnosql.documents.azure.com:443/",
                //    tokenCredential: DefaultAzureCredentialHelper.GetDefaultAzureCredential(AzureAuthorityHosts.AzurePublicCloud.AbsoluteUri),
                //    new CosmosClientOptions()
                //    {
                //        // When initializing CosmosClient manually, setting this property is required 
                //        // due to limitations in default serializer. 
                //        UseSystemTextJsonSerializerWithOptions = JsonSerializerOptions.Default,
                //    });
                //await cosmosClient.CreateDatabaseAsync("antragdb");
                //var database = cosmosClient.GetDatabase("kuduaidb");
                //var container = database.GetContainer("pdfcontent");
                //await container.DeleteContainerAsync();
                //var collection = await database.CreateContainerIfNotExistsAsync("pdfcontent", "/Key");
                //return cosmosClient.GetDatabase(appConfig.AzureCosmosDBNoSQLConfig.DatabaseName);

                var action = args.Length == 0 ? "Usage" : args[0];
                if (!_methods.TryGetValue(action, out var func))
                {

                    Console.WriteLine($"Action '{action}' is not supported");
                    await Usage(args);
                    return;
                }

                await func(args);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex);
            }
        }

        static async Task Assist(string[] args)
        {
            if (args.Length != 2)
            {
                Console.WriteLine("ChatCompletionWithRAG.exe Assist category");
                return;
            }

            var category = args[1];
            var kernel = CreateKernelWithPlugins(category);
            //var vectorStore = kernel.GetRequiredService<IVectorStore>();
            var collection = kernel.GetRequiredService<IVectorStoreRecordCollection<string, RAGFileInfo>>();
            if (collection is InMemoryVectorStoreRecordCollection<string, RAGFileInfo>)
            {
                throw new NotSupportedException("Assist is not supported for InMemoryVectorStore");
            }

            var textEmbeddingService = kernel.GetRequiredService<ITextEmbeddingGenerationService>();
            kernel.Plugins.AddFromObject(new TextSearchProvider(collection, kernel, textEmbeddingService), pluginName: "TextSearchProvider");

            //await RunChatLoop(kernel);
            await RunChatLoopWithTemplate(kernel);
        }

        static async Task Learn(string[] args)
        {
            if (args.Length < 2)
            {
                Console.WriteLine("ChatCompletionWithRAG.exe Learn category directory [*.txt,*.md]");
                return;
            }

            var category = args[1];
            var directory = args[2];
            var patterns = args.Length > 3 ? args[3] : DefaultFileExtensions;

            var kernel = CreateKernelWithPlugins(category);
            //var vectorStore = kernel.GetRequiredService<IVectorStore>();
            //var collection = vectorStore.GetCollection<string, RAGFileInfo>(category);
            //await collection.CreateCollectionIfNotExistsAsync();

            var collection = kernel.GetRequiredService<IVectorStoreRecordCollection<string, RAGFileInfo>>();
            await collection.CreateCollectionIfNotExistsAsync();

            // Create and upsert glossary entries into the collection.
            var textEmbeddingService = kernel.GetRequiredService<ITextEmbeddingGenerationService>();
            var ragFiles = await LearnAndUpsertAsync(collection, directory, patterns, kernel, textEmbeddingService).ConfigureAwait(false);
            // Testing Cache
            // await LearnAndUpsertAsync(collection, directory, patterns, kernel, textEmbeddingService).ConfigureAwait(false);

            if (collection is InMemoryVectorStoreRecordCollection<string, RAGFileInfo>)
            {
                kernel.Plugins.AddFromObject(new LocalFileSearchProvider(ragFiles, kernel, textEmbeddingService), pluginName: nameof(LocalFileSearchProvider));
                kernel.Plugins.AddFromObject(new TextSearchProvider(collection, kernel, textEmbeddingService), pluginName: nameof(TextSearchProvider));

                //await RunChatLoop(kernel);
                await RunChatLoopWithTemplate(kernel);
            }
        }

        static async Task<List<RAGFileInfo>> LearnAndUpsertAsync(IVectorStoreRecordCollection<string, RAGFileInfo> collection, string directory, string patterns, Kernel kernel, ITextEmbeddingGenerationService textEmbeddingService)
        {
            var ragFiles = new List<RAGFileInfo>();
            var startIndex = 0;
            var batchSize = 3;
            await foreach (var ragFile in LearnFromDirectoryAsync(collection, directory, patterns, kernel, textEmbeddingService))
            {
                ragFiles.Add(ragFile);
                if (ragFiles.Count % batchSize == 0)
                {
                    //await foreach (var result in collection.UpsertBatchAsync(ragFiles.Skip(startIndex)))
                    //{
                    //    var item = ragFiles.Find(r => r.Key == result);
                    //    Console.WriteLine($"Upsert: {result}, File: {item.FullName}#{item.ChunkIndex:000}");
                    //}
                    await collection.UpsertBatchAsync(ragFiles.Skip(startIndex)).ToListAsync();
                    startIndex = ragFiles.Count;
                }
            }
            if (ragFiles.Count > startIndex)
            {
                //await foreach (var result in collection.UpsertBatchAsync(ragFiles.Skip(startIndex)))
                //{
                //    var item = ragFiles.Find(r => r.Key == result);
                //    //Console.WriteLine($"Upsert: {result}, File: {item.FullName}#{item.ChunkIndex:000}");
                //}
                await collection.UpsertBatchAsync(ragFiles.Skip(startIndex)).ToListAsync();
            }

            return ragFiles;
            //var ragFiles = await LearnFromDirectoryAsync(directory, patterns, kernel, textEmbeddingService);
            // need to chunk
            //await collection.UpsertBatchAsync(ragFiles).ToListAsync();
        }

        static Task CleanCache(string[] args)
        {
            Directory.Delete(Program.RAGCacheDirectory, recursive: true);
            Console.WriteLine($"{Program.RAGCacheDirectory} deleted");
            return Task.CompletedTask;
        }

        static Task Usage(string[] args)
        {
            Console.WriteLine("Usage:");
            Console.WriteLine("ChatCompletionWithRAG.exe Learn category directory [*.txt,*.md]");
            Console.WriteLine("ChatCompletionWithRAG.exe CleanCache");
            Console.WriteLine("ChatCompletionWithRAG.exe Assist category");
            return Task.CompletedTask;
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

        static async Task RunChatLoopWithTemplate(Kernel kernel)
        {
            var chatClient = kernel.GetRequiredService<IChatCompletionService>();
            var alternativeQuestionsTemplate = """
                You are an AI language model assistant. 
                Your task is to generate 5 different versions of the given user question to retrieve relevant documents from a vector database.
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
                {{LocalFileSearchProvider.GetInformation $query}}.
                Do include source references in your answer as footnotes.
                Question: {{$originalQuestion}}
                """;

            var promptSearchResultsTemplate = """
                Please use this information to answer the question:
                {{#with (TextSearchProvider-GetTextSearchResults query)}}  
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
                var originalQuestion = "How long is MSI token cached?"; // Console.ReadLine();
                if (string.IsNullOrWhiteSpace(originalQuestion))
                {
                    break;
                }
                Console.WriteLine(originalQuestion);

                var result = await kernel.InvokePromptAsync(
                    promptTemplate: alternativeQuestionsTemplate,
                    arguments: new(executionSettings) { { "originalQuestion", originalQuestion } })
                    .WithProgress().ConfigureAwait(false);

                var questionsResult = JsonSerializer.Deserialize<QuestionsResult>(result.ToString());
                //for (int i = 0; i < questionsResult.AlternativeQuestions.Length; i++)
                //{
                //    Console.WriteLine($"Alternative Question[{i}]: {questionsResult.AlternativeQuestions[i]}");
                //}

                var useLocalFile = false;
                if (useLocalFile)
                {
                    result = await kernel.InvokePromptAsync(
                        promptTemplate: promptTemplate,
                        arguments: new() { { "originalQuestion", originalQuestion }, { "query", result } })
                        .WithProgress().ConfigureAwait(false);
                    Console.WriteLine($"Answer: {result}");
                }
                else
                {
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
                }

                Console.WriteLine();
                break;
            }
        }

//        static async IAsyncEnumerable<RAGFileInfo> LearnFromDirectoryExAsync(string directory, string patterns, Kernel kernel, ITextEmbeddingGenerationService textEmbeddingService)
//        {
//            Console.WriteLine($"Learning from {directory} with patterns {patterns}");
//            Console.WriteLine($"Caching at {Program.RAGCacheDirectory}");
//            var results = new List<RAGFileInfo>();
//            foreach (var pattern in patterns.Split([',', ';']))
//            {
//                await foreach (var ragFile in RAGFileInfo.EnumerateDirectory(directory, pattern, kernel, textEmbeddingService))
//                {
//                    yield return ragFile;
//                }
//            }
////            return results;
//        }

        static async IAsyncEnumerable<RAGFileInfo> LearnFromDirectoryAsync(IVectorStoreRecordCollection<string, RAGFileInfo> collection, string directory, string patterns, Kernel kernel, ITextEmbeddingGenerationService textEmbeddingService)
        {
            Console.WriteLine($"Learning from {directory} with patterns {patterns}");
            Console.WriteLine($"Caching at {Program.RAGCacheDirectory}");
            var results = new List<RAGFileInfo>();
            foreach (var pattern in patterns.Split([',', ';']))
            {
                await foreach (var ragFile in RAGFileInfo.EnumerateDirectory(collection, directory, pattern, kernel, textEmbeddingService))
                {
                    yield return ragFile;
                }
            }
            //return results;
        }

        private static Kernel CreateKernelWithPlugins(string category, params Type[] pluginTypes)
        {
            var clientOptions = new AzureOpenAIClientOptions();
            if (HttpLoggingTraceLevel != TraceLevel.Off)
            {
                clientOptions.Transport = new HttpClientPipelineTransport(new HttpClient(new HttpLoggingHandler(traceLevel: HttpLoggingTraceLevel)));
            }

            var customClient = new AzureOpenAIClient(
                new Uri($"https://{AzureAIName}.openai.azure.com/"),
                DefaultAzureCredentialHelper.GetDefaultAzureCredential(AzureAuthorityHosts.AzurePublicCloud.AbsoluteUri),
                clientOptions);

            // Create a kernel with OpenAI chat completion and WeatherPlugin
            var builder = Kernel.CreateBuilder();
            builder.AddAzureOpenAIChatCompletion(AzureAIDeploymentName, customClient);
            builder.AddAzureOpenAITextEmbeddingGeneration(AzureAITextEmbeddingDeploymentName, azureOpenAIClient: customClient);

            // VectorStore
            var useInMemoryVectorStore = false;
            if (useInMemoryVectorStore)
            {
                // builder.AddInMemoryVectorStore();
                builder.AddInMemoryVectorStoreRecordCollection<string, RAGFileInfo>(category);
            }
            else
            {
                builder.Services.AddSingleton<Database>(
                    sp => new CosmosClient(
                            accountEndpoint: "https://kuduaicosmosdbnosql.documents.azure.com:443/",
                            tokenCredential: DefaultAzureCredentialHelper.GetDefaultAzureCredential(AzureAuthorityHosts.AzurePublicCloud.AbsoluteUri), // new AzureCliCredential(),
                            new CosmosClientOptions()
                            {
                                // When initializing CosmosClient manually, setting this property is required 
                                // due to limitations in default serializer. 
                                UseSystemTextJsonSerializerWithOptions = JsonSerializerOptions.Default,
                                HttpClientFactory = () => new HttpClient(new HttpLoggingHandler(traceLevel: HttpLoggingTraceLevel)),
                            }).GetDatabase(VectorDatabaseName));
                // GetCollections not working - Only String and AzureCosmosDBNoSQLCompositeKey keys are supported
                // builder.AddAzureCosmosDBNoSQLVectorStore();
                builder.AddAzureCosmosDBNoSQLVectorStoreRecordCollection<RAGFileInfo>(category);
            }

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
            Program.Write(info);
            while (task != await Task.WhenAny(Task.Delay(dotIntercalSecs * 1_000), task).ConfigureAwait(false))
            {
                Console.Write(".");
            }
            Console.WriteLine();

            return await task.ConfigureAwait(false);
        }

        internal static void WriteLine(object msg)
        {
            WriteLine("{0}", msg);
        }

        internal static void WriteLine(string format, params object[] args)
        {
            if (DebugTraceLevel != TraceLevel.Off)
            {
                Console.WriteLine(format, args);
            }
        }

        internal static void Write(string msg)
        {
            if (DebugTraceLevel != TraceLevel.Off)
            {
                Console.Write("{0}", msg);
            }
        }
    }
}
