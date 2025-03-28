﻿// Copyright (c) Microsoft. All rights reserved.

using System.Reflection;
using System.Runtime.CompilerServices;
using Microsoft.Extensions.VectorData;
using Microsoft.SemanticKernel.Embeddings;

namespace ChatCompletionWithRAG;

/// <summary>
/// An attribute that can be used for an embedding property to indicate that it should
/// be generated from one or more text properties located on the same class.
/// </summary>
/// <remarks>
/// This class is part of the <see cref="VectorStore_EmbeddingGeneration"/> sample.
/// </remarks>
[AttributeUsage(AttributeTargets.Property, AllowMultiple = false, Inherited = true)]
public sealed class TextEmbeddingAttribute : Attribute
{
    /// <summary>
    /// Initializes a new instance of the <see cref="TextEmbeddingAttribute"/> class.
    /// </summary>
    /// <param name="sourcePropertyName">The name of the property that the embedding should be generated from.</param>
#pragma warning disable CA1019 // Define accessors for attribute arguments
    public TextEmbeddingAttribute(string sourcePropertyName)
#pragma warning restore CA1019 // Define accessors for attribute arguments
    {
        this.SourcePropertyNames = [sourcePropertyName];
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="TextEmbeddingAttribute"/> class.
    /// </summary>
    /// <param name="sourcePropertyNames">The names of the properties that the embedding should be generated from.</param>
    public TextEmbeddingAttribute(string[] sourcePropertyNames)
    {
        this.SourcePropertyNames = sourcePropertyNames;
    }

    /// <summary>
    /// Gets the name of the property to use as the source for generating the embedding.
    /// </summary>
    public string[] SourcePropertyNames { get; }
}

/// <summary>
/// Decorator for a <see cref="IVectorStore"/> that generates embeddings for records on upsert.
/// </summary>
/// <remarks>
/// This class is part of the <see cref="VectorStore_EmbeddingGeneration"/> sample.
/// </remarks>
public class TextEmbeddingVectorStore : IVectorStore
{
    /// <summary>The decorated <see cref="IVectorStore"/>.</summary>
    private readonly IVectorStore _innerVectorStore;

    /// <summary>The service to use for generating the embeddings.</summary>
    private readonly ITextEmbeddingGenerationService _textEmbeddingGenerationService;

    /// <summary>
    /// Initializes a new instance of the <see cref="TextEmbeddingVectorStore"/> class.
    /// </summary>
    /// <param name="decoratedVectorStore">The decorated <see cref="IVectorStore"/>.</param>
    /// <param name="textEmbeddingGenerationService">The service to use for generating the embeddings.</param>
    public TextEmbeddingVectorStore(IVectorStore decoratedVectorStore, ITextEmbeddingGenerationService textEmbeddingGenerationService)
    {
        // Verify & Assign.
        this._innerVectorStore = decoratedVectorStore ?? throw new ArgumentNullException(nameof(decoratedVectorStore));
        this._textEmbeddingGenerationService = textEmbeddingGenerationService ?? throw new ArgumentNullException(nameof(textEmbeddingGenerationService));
    }

    /// <inheritdoc />
    public IVectorStoreRecordCollection<TKey, TRecord> GetCollection<TKey, TRecord>(string name, VectorStoreRecordDefinition? vectorStoreRecordDefinition = null)
        where TKey : notnull
    {
        var collection = this._innerVectorStore.GetCollection<TKey, TRecord>(name, vectorStoreRecordDefinition);
        var embeddingStore = new TextEmbeddingVectorStoreRecordCollection<TKey, TRecord>(collection, this._textEmbeddingGenerationService);
        return embeddingStore;
    }

    /// <inheritdoc />
    public IAsyncEnumerable<string> ListCollectionNamesAsync(CancellationToken cancellationToken = default)
    {
        return this._innerVectorStore.ListCollectionNamesAsync(cancellationToken);
    }

    /// <summary>
    /// Decorator for a <see cref="IVectorStoreRecordCollection{TKey, TRecord}"/> that generates embeddings for records on upsert and when using <see cref="VectorizableTextSearchAsync(string, VectorSearchOptions?, CancellationToken)"/>.
    /// </summary>
    /// <remarks>
    /// This class is part of the <see cref="VectorStore_EmbeddingGeneration"/> sample.
    /// </remarks>
    /// <typeparam name="TKey">The data type of the record key.</typeparam>
    /// <typeparam name="TRecord">The record data model to use for adding, updating and retrieving data from the store.</typeparam>
#pragma warning disable CA1711 // Identifiers should not have incorrect suffix
    public class TextEmbeddingVectorStoreRecordCollection<TKey, TRecord> : IVectorStoreRecordCollection<TKey, TRecord>, IVectorizableTextSearch<TRecord>
#pragma warning restore CA1711 // Identifiers should not have incorrect suffix
        where TKey : notnull
    {
        /// <summary>The decorated <see cref="IVectorStoreRecordCollection{TKey, TRecord}"/>.</summary>
        private readonly IVectorStoreRecordCollection<TKey, TRecord> _innerVectorStoreRecordCollection;

        /// <summary>The service to use for generating the embeddings.</summary>
        private readonly ITextEmbeddingGenerationService _textEmbeddingGenerationService;

        /// <summary>Optional configuration options for this class.</summary>
        private readonly IEnumerable<(PropertyInfo EmbeddingPropertyInfo, IList<PropertyInfo> SourcePropertiesInfo)> _embeddingPropertiesWithSourceProperties;

        /// <summary>
        /// Initializes a new instance of the <see cref="TextEmbeddingVectorStoreRecordCollection{TKey, TRecord}"/> class.
        /// </summary>
        /// <param name="decoratedVectorStoreRecordCollection">The decorated <see cref="IVectorStoreRecordCollection{TKey, TRecord}"/>.</param>
        /// <param name="textEmbeddingGenerationService">The service to use for generating the embeddings.</param>
        /// <exception cref="ArgumentException">Thrown when embedding properties are referencing data source properties that do not exist.</exception>
        /// <exception cref="ArgumentNullException">Thrown when required parameters are null.</exception>
        public TextEmbeddingVectorStoreRecordCollection(IVectorStoreRecordCollection<TKey, TRecord> decoratedVectorStoreRecordCollection, ITextEmbeddingGenerationService textEmbeddingGenerationService)
        {
            // Assign.
            this._innerVectorStoreRecordCollection = decoratedVectorStoreRecordCollection ?? throw new ArgumentNullException(nameof(decoratedVectorStoreRecordCollection));
            this._textEmbeddingGenerationService = textEmbeddingGenerationService ?? throw new ArgumentNullException(nameof(textEmbeddingGenerationService));

            // Find all the embedding properties to generate embeddings for.
            this._embeddingPropertiesWithSourceProperties = FindDataPropertiesWithEmbeddingProperties(typeof(TRecord));
        }

        /// <inheritdoc />
        public string CollectionName => this._innerVectorStoreRecordCollection.CollectionName;

        /// <inheritdoc />
        public Task<bool> CollectionExistsAsync(CancellationToken cancellationToken = default)
        {
            return this._innerVectorStoreRecordCollection.CollectionExistsAsync(cancellationToken);
        }

        /// <inheritdoc />
        public Task CreateCollectionAsync(CancellationToken cancellationToken = default)
        {
            return this._innerVectorStoreRecordCollection.CreateCollectionAsync(cancellationToken);
        }

        /// <inheritdoc />
        public async Task CreateCollectionIfNotExistsAsync(CancellationToken cancellationToken = default)
        {
            if (!await this.CollectionExistsAsync(cancellationToken).ConfigureAwait(false))
            {
                await this.CreateCollectionAsync(cancellationToken).ConfigureAwait(false);
            }
        }

        /// <inheritdoc />
        public Task DeleteCollectionAsync(CancellationToken cancellationToken = default)
        {
            return this._innerVectorStoreRecordCollection.DeleteCollectionAsync(cancellationToken);
        }

        /// <inheritdoc />
        public Task DeleteAsync(TKey key, DeleteRecordOptions? options = null, CancellationToken cancellationToken = default)
        {
            return this._innerVectorStoreRecordCollection.DeleteAsync(key, options, cancellationToken);
        }

        /// <inheritdoc />
        public Task DeleteBatchAsync(IEnumerable<TKey> keys, DeleteRecordOptions? options = null, CancellationToken cancellationToken = default)
        {
            return this._innerVectorStoreRecordCollection.DeleteBatchAsync(keys, options, cancellationToken);
        }

        /// <inheritdoc />
        public Task<TRecord?> GetAsync(TKey key, GetRecordOptions? options = null, CancellationToken cancellationToken = default)
        {
            return this._innerVectorStoreRecordCollection.GetAsync(key, options, cancellationToken);
        }

        /// <inheritdoc />
        public IAsyncEnumerable<TRecord> GetBatchAsync(IEnumerable<TKey> keys, GetRecordOptions? options = null, CancellationToken cancellationToken = default)
        {
            return this._innerVectorStoreRecordCollection.GetBatchAsync(keys, options, cancellationToken);
        }

        /// <inheritdoc />
        public async Task<TKey> UpsertAsync(TRecord record, UpsertRecordOptions? options = null, CancellationToken cancellationToken = default)
        {
            var recordWithEmbeddings = await this.AddEmbeddingsAsync(record, cancellationToken).ConfigureAwait(false);
            return await this._innerVectorStoreRecordCollection.UpsertAsync(recordWithEmbeddings, options, cancellationToken).ConfigureAwait(false);
        }

        /// <inheritdoc />
        public async IAsyncEnumerable<TKey> UpsertBatchAsync(IEnumerable<TRecord> records, UpsertRecordOptions? options = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var recordWithEmbeddingsTasks = records.Select(r => this.AddEmbeddingsAsync(r, cancellationToken));
            var recordWithEmbeddings = await Task.WhenAll(recordWithEmbeddingsTasks).ConfigureAwait(false);
            var upsertResults = this._innerVectorStoreRecordCollection.UpsertBatchAsync(recordWithEmbeddings, options, cancellationToken);
            await foreach (var upsertResult in upsertResults.ConfigureAwait(false))
            {
                yield return upsertResult;
            }
        }

        /// <inheritdoc />
        public Task<VectorSearchResults<TRecord>> VectorizedSearchAsync<TVector>(TVector vector, VectorSearchOptions? options = null, CancellationToken cancellationToken = default)
        {
            return this._innerVectorStoreRecordCollection.VectorizedSearchAsync(vector, options, cancellationToken);
        }

        /// <inheritdoc />
        public async Task<VectorSearchResults<TRecord>> VectorizableTextSearchAsync(string searchText, VectorSearchOptions? options = null, CancellationToken cancellationToken = default)
        {
            var embeddingValue = await this._textEmbeddingGenerationService.GenerateEmbeddingAsync(searchText, cancellationToken: cancellationToken).ConfigureAwait(false);
            return await this.VectorizedSearchAsync(embeddingValue, options, cancellationToken).ConfigureAwait(false);
        }

        /// <summary>
        /// Generate and add embeddings for each embedding field that has a <see cref="GenerateTextEmbeddingAttribute"/> on the provided record.
        /// </summary>
        /// <param name="record">The record to generate embeddings for.</param>
        /// <param name="cancellationToken">The <see cref="CancellationToken"/> to monitor for cancellation requests.</param>
        /// <returns>The record with embeddings added.</returns>
        private async Task<TRecord> AddEmbeddingsAsync(TRecord record, CancellationToken cancellationToken)
        {
            foreach (var (embeddingPropertyInfo, sourcePropertiesInfo) in this._embeddingPropertiesWithSourceProperties)
            {
                var sourceValues = sourcePropertiesInfo.Select(x => x.GetValue(record)).Cast<string>().Where(x => !string.IsNullOrWhiteSpace(x));
                var sourceString = string.Join("\n", sourceValues);

                var embeddingValue = await this._textEmbeddingGenerationService.GenerateEmbeddingAsync(sourceString, cancellationToken: cancellationToken).ConfigureAwait(false);
                embeddingPropertyInfo.SetValue(record, embeddingValue);
            }

            return record;
        }

        /// <summary>
        /// Get the list of properties with <see cref="GenerateTextEmbeddingAttribute"/> from the data model.
        /// </summary>
        /// <param name="dataModelType">The type of the data model to find </param>
        /// <returns>The list of properties with <see cref="GenerateTextEmbeddingAttribute"/> with the properties from which the embedding can be generated.</returns>
        private static IEnumerable<(PropertyInfo EmbeddingPropertyInfo, IList<PropertyInfo> SourcePropertiesInfo)> FindDataPropertiesWithEmbeddingProperties(Type dataModelType)
        {
            var allProperties = dataModelType.GetProperties();
            var propertiesDictionary = allProperties.ToDictionary(p => p.Name);

            // Loop through all the properties to find the ones that have the GenerateTextEmbeddingAttribute.
            foreach (var property in allProperties)
            {
                var attribute = property.GetCustomAttribute<TextEmbeddingAttribute>();
                if (attribute is not null)
                {
                    // Find the source properties that the embedding should be generated from.
                    var sourcePropertiesInfo = new List<PropertyInfo>();
                    foreach (var sourcePropertyName in attribute.SourcePropertyNames)
                    {
                        if (!propertiesDictionary.TryGetValue(sourcePropertyName, out var sourcePropertyInfo))
                        {
                            throw new ArgumentException($"The source property '{sourcePropertyName}' as referenced by embedding property '{property.Name}' does not exist in the record model.");
                        }
                        else if (sourcePropertyInfo.PropertyType != typeof(string))
                        {
                            throw new ArgumentException($"The source property '{sourcePropertyName}' as referenced by embedding property '{property.Name}' has type {sourcePropertyInfo.PropertyType} but must be a string.");
                        }
                        else
                        {
                            sourcePropertiesInfo.Add(sourcePropertyInfo);
                        }
                    }

                    yield return (property, sourcePropertiesInfo);
                }
            }
        }
    }
}
