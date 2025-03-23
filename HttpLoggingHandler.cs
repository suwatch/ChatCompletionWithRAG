using System.Diagnostics;
using System.Text.Json;

namespace ChatCompletionWithRAG
{
    public sealed class HttpLoggingHandler : DelegatingHandler
    {
        private readonly bool _verbose;

        public HttpLoggingHandler(HttpMessageHandler innerHandler = null, bool verbose = true)
            : base(innerHandler ?? new HttpClientHandler())
        {
            _verbose = verbose;
        }

#if NET6_0_OR_GREATER
        protected override HttpResponseMessage Send(HttpRequestMessage request, CancellationToken cancellationToken)
            => SendAsync(request, cancellationToken, async: false).GetAwaiter().GetResult();
#endif
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
            => SendAsync(request, cancellationToken, async: true);

        private async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken, bool async)
        {
            WriteLine("---------- Request -----------------------");
            WriteLine($"{request.Method} {request.RequestUri.PathAndQuery} HTTP/{request.Version}");
            Write("Host: ");
            WriteLine(request.RequestUri.Host);
            foreach (var header in request.Headers)
            {
                Write($"{header.Key}: ");
                if (new[] { "Authorization", "api-key" }.Any(k => string.Equals(k, header.Key)))
                {
                    WriteLine(header.Value.First().Substring(0, 20) + "...");
                }
                else
                {
                    WriteLine(string.Join("; ", header.Value));
                }
            }
            if (request.Content is not null)
            {
                foreach (var header in request.Content.Headers)
                {
                    WriteLine($"{header.Key}: {string.Join("; ", header.Value)}");
                }
            }

            WriteLine();

            if (_verbose && request.Content is not null)
            {
                request.Content = await DumpContent(request.Content, cancellationToken, async).ConfigureAwait(false);
                WriteLine();
            }

            var watch = new Stopwatch();
            watch.Start();
#if NET
            HttpResponseMessage response = async ? (await base.SendAsync(request, cancellationToken).ConfigureAwait(false)) : base.Send(request, cancellationToken);
#else
            HttpResponseMessage response = await base.SendAsync(request, cancellationToken).ConfigureAwait(false);
#endif
            watch.Stop();

            WriteLine($"HTTP/{response.Version} {(int)response.StatusCode} {response.StatusCode}");

            foreach (var header in response.Headers)
            {
                Write($"{header.Key}: ");
                WriteLine(string.Join("; ", header.Value));
            }
            if (response.Content is not null)
            {
                foreach (var header in response.Content.Headers)
                {
                    Write($"{header.Key}: ");
                    WriteLine(string.Join("; ", header.Value));
                }
            }
            WriteLine();

            if (_verbose && response.Content is not null)
            {
                response.Content = await DumpContent(response.Content, cancellationToken, async).ConfigureAwait(false);
                WriteLine();
            }

            WriteLine($"---------- Done ({(int)response.StatusCode} {response.StatusCode}, {watch.ElapsedMilliseconds} ms, request: {request.Content?.Headers.ContentLength} bytes) ------------");
            return response;
        }

        private async Task<HttpContent> DumpContent(HttpContent content, CancellationToken cancellationToken, bool async)
        {
            if (content == null || content.Headers.ContentType == null)
            {
                return content;
            }

            (var clone, var ms) = await ReadAndCloneHttpContentAsync(content, cancellationToken, async).ConfigureAwait(false);
            var reader = new StreamReader(ms);
#if NET
            var result = async ? await reader.ReadToEndAsync(cancellationToken).ConfigureAwait(false) : reader.ReadToEnd();
#else
            var result = reader.ReadToEnd();
#endif

            if (content.Headers.ContentType.MediaType == "application/json")
            {
                using var jDoc = JsonDocument.Parse(result);
                Write(JsonSerializer.Serialize(jDoc, new JsonSerializerOptions { WriteIndented = true }));
            }
            else
            {
                WriteLine(result);
            }

            ms.Position = 0;
            return clone;
        }

        public static async Task<(HttpContent, MemoryStream)> ReadAndCloneHttpContentAsync(HttpContent content, CancellationToken cancellationToken, bool async)
        {
            if (content is null)
            {
                return (null, null);
            }

            MemoryStream ms = new MemoryStream();
#if NET
            if (async)
            {
                await content.CopyToAsync(ms, cancellationToken).ConfigureAwait(false);
            }
            else
            {
                content.CopyTo(ms, default, cancellationToken);
            }
#else
            await content.CopyToAsync(ms).ConfigureAwait(false);
#endif

            ms.Position = 0;
            var clone = new StreamContent(ms);

            // Copy the content headers
            foreach (var h in content.Headers)
            {
                clone.Headers.TryAddWithoutValidation(h.Key, h.Value);
            }

            return (clone, ms);
        }

        private static void WriteLine(string message = null)
        {
            Console.WriteLine(message);
            Debug.WriteLine(message);
        }

        private static void Write(string message = null)
        {
            Console.Write(message);
            Debug.Write(message);
        }
    }
}
