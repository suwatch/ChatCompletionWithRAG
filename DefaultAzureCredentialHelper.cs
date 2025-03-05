using System.Collections.Concurrent;
using System.Diagnostics;
using System.Security.Cryptography.X509Certificates;
using Azure.Core;
using Azure.Identity;

namespace ChatCompletionWithRAG
{
    public static class DefaultAzureCredentialHelper
    {
        private const string TokenCacheName = "ANTARES_TOKEN_CACHE";
        private static readonly ConcurrentDictionary<string, AccessTokenCache> _inMemoryTokenCaches = new(StringComparer.OrdinalIgnoreCase);
        private static readonly ConcurrentDictionary<string, TokenCredential> _defaultAzureCredentials = new(StringComparer.OrdinalIgnoreCase);

        public static AccessToken GetUserToken(string authorityHost, string tenantId, string scope, CancellationToken cancellationToken = default)
            => _inMemoryTokenCaches.GetOrAdd(GetCacheKey("user", tenantId, scope), () => GetDefaultAzureCredential(authorityHost, null).GetToken(new TokenRequestContext(
                scopes: [scope],
                tenantId: tenantId,
                parentRequestId: GetActivityId()
            ), cancellationToken));

        public static async ValueTask<AccessToken> GetUserTokenAsync(string authorityHost, string tenantId, string scope, CancellationToken cancellationToken = default)
            => await _inMemoryTokenCaches.GetOrAddAsync(GetCacheKey("user", tenantId, scope), async () => await GetDefaultAzureCredential(authorityHost, null).GetTokenAsync(new TokenRequestContext(
                scopes: [scope],
                tenantId: tenantId,
                parentRequestId: GetActivityId()
            ), cancellationToken)).ConfigureAwait(false);

        public static AccessToken GetManagedIdentityToken(string authorityHost, string managedIdentityResourceId, string tenantId, string scope, CancellationToken cancellationToken = default)
            => _inMemoryTokenCaches.GetOrAdd(GetCacheKey(managedIdentityResourceId, tenantId, scope), () => GetDefaultAzureCredential(authorityHost, managedIdentityResourceId).GetToken(new TokenRequestContext(
                scopes: [scope],
                tenantId: tenantId,
                parentRequestId: GetActivityId()
            ), cancellationToken));

        public static async ValueTask<AccessToken> GetManagedIdentityTokenAsync(string authorityHost, string managedIdentityResourceId, string tenantId, string scope, CancellationToken cancellationToken = default)
            => await _inMemoryTokenCaches.GetOrAddAsync(GetCacheKey(managedIdentityResourceId, tenantId, scope), () => GetDefaultAzureCredential(authorityHost, managedIdentityResourceId).GetTokenAsync(new TokenRequestContext(
                scopes: [scope],
                tenantId: tenantId,
                parentRequestId: GetActivityId()
            ), cancellationToken)).ConfigureAwait(false);

        public static DefaultAzureCredential GetDefaultAzureCredential(string authorityHost, string managedIdentityResourceId = null)
            => (DefaultAzureCredential)_defaultAzureCredentials.GetOrAdd(managedIdentityResourceId ?? string.Empty, id =>
            {
                var isInteractive = string.IsNullOrEmpty(id);
                var options = new DefaultAzureCredentialOptions
                {
                    AuthorityHost = new UriBuilder(authorityHost) { Path = "/" }.Uri,
                    ExcludeInteractiveBrowserCredential = !isInteractive,
                    // For MI, token is cached by default, see https://devblogs.microsoft.com/azure-sdk/azure-sdk-release-august-2022/
                    ExcludeManagedIdentityCredential = isInteractive,
                    // file-based token cache only applicable for user token
                    // such as C:\Users\suwatch\AppData\Local.IdentityService\msal.cache.nocae
                    ExcludeSharedTokenCacheCredential = !isInteractive,
                    // exclude other credential sources
                    // preventing DefaultAzureCredential to iterate getting token for all credential soruces
                    ExcludeAzureCliCredential = true,
                    ExcludeVisualStudioCodeCredential = true,
                    ExcludeVisualStudioCredential = true,
                    ExcludeEnvironmentCredential = true,
                    ExcludeAzureDeveloperCliCredential = true,
                    ExcludeAzurePowerShellCredential = true,
                    ExcludeWorkloadIdentityCredential = true,
                };

                if (!isInteractive)
                {
                    // for system-assigned, both ManagedIdentityClientId and ManagedIdentityResourceId MUST not be set.
                    if (Guid.TryParse(id, out _))
                    {
                        options.ManagedIdentityClientId = id;
                    }
                    else if (managedIdentityResourceId.StartsWith("/subscriptions/", StringComparison.OrdinalIgnoreCase))
                    {
                        options.ManagedIdentityResourceId = new(id);
                    }
                }

                return new DefaultAzureCredential(options);
            });

        public static ClientCertificateCredential GetClientCertificateCredential(string authorityHost, string tenantId, string clientId, X509Certificate2 clientCertificate)
            => (ClientCertificateCredential)_defaultAzureCredentials.GetOrAdd($"{tenantId}_{clientId}_{clientCertificate.Thumbprint}", id =>
            {
                var options = new ClientCertificateCredentialOptions
                {
                    AuthorityHost = new UriBuilder(authorityHost) { Path = "/" }.Uri,
                    SendCertificateChain = true,
                    // https://github.com/Azure/azure-sdk-for-net/blob/Azure.Identity_1.13.1/sdk/identity/Azure.Identity/samples/TokenCache.md
                    // By setting UnsafeAllowUnencryptedStorage to true, the credential will encrypt the contents of the token cache before persisting it if data protection is available on the current platform.
                    // If platform data protection is unavailable, it will write and read the persisted token data to an unencrypted local file ACL'd to the current account.
                    // If UnsafeAllowUnencryptedStorage is false (the default), a CredentialUnavailableException will be raised in the case no data protection is available.
                    TokenCachePersistenceOptions = new()
                    {
                        Name = TokenCacheName,
                        UnsafeAllowUnencryptedStorage = true,
                    },
                };

                return new ClientCertificateCredential(tenantId: tenantId, clientId: clientId, clientCertificate: clientCertificate, options: options);
            });

        private static string GetActivityId()
            => $"{(Trace.CorrelationManager.ActivityId == Guid.Empty ? Guid.NewGuid() : Trace.CorrelationManager.ActivityId)}";

        private static string GetCacheKey(string managedIdentityResourceId, string tenantId, string scope)
            => $"{managedIdentityResourceId}_{tenantId}_{scope}";

        private static AccessToken GetOrAdd(this ConcurrentDictionary<string, AccessTokenCache> caches, string cacheKey, Func<AccessToken> func)
        {
            // implement in-memory token cache to provide consistency across all credential sources
            if (caches.TryGetValue(cacheKey, out var cache) && cache.ExpiresOn > DateTimeOffset.UtcNow)
            {
                return cache.Token;
            }

            try
            {
                var accessToken = func();
                caches[cacheKey] = new() { Token = accessToken, ExpiresOn = GetCacheExpiresOn(accessToken.ExpiresOn) };
                return accessToken;
            }
            catch (Exception)
            {
                // best effort to continue to use valid token
                if (cache != null && cache.Token.ExpiresOn > DateTimeOffset.UtcNow.AddMinutes(1))
                {
                    return cache.Token;
                }

                throw;
            }
        }

        private static async ValueTask<AccessToken> GetOrAddAsync(this ConcurrentDictionary<string, AccessTokenCache> caches, string cacheKey, Func<ValueTask<AccessToken>> funcTask)
        {
            // implement in-memory token cache to provide consistency across all credential sources
            if (caches.TryGetValue(cacheKey, out var cache) && cache.ExpiresOn > DateTimeOffset.UtcNow)
            {
                return cache.Token;
            }

            try
            {
                var accessToken = await funcTask().ConfigureAwait(false);
                caches[cacheKey] = new() { Token = accessToken, ExpiresOn = GetCacheExpiresOn(accessToken.ExpiresOn) };
                return accessToken;
            }
            catch (Exception)
            {
                // best effort to continue to use valid token
                if (cache != null && cache.Token.ExpiresOn > DateTimeOffset.UtcNow.AddMinutes(1))
                {
                    return cache.Token;
                }

                throw;
            }
        }

        // cache only half of the token life time
        private static DateTimeOffset GetCacheExpiresOn(DateTimeOffset expiresOn)
            => expiresOn.Subtract(TimeSpan.FromSeconds((expiresOn - DateTimeOffset.UtcNow).TotalSeconds / 2));

        private sealed class AccessTokenCache
        {
            public AccessToken Token { get; set; }
            public DateTimeOffset ExpiresOn { get; set; }
        }
    }
}
