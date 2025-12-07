#---- Libraries ----
library(tidyverse)
library(reticulate)
library(glue)

# Init flag indicates if initialization should run.
# If FALSE, load needed files from CSV.
init_flag <- FALSE

# Embed flag indicates if chunks need creating and embedding.
# If FALSE, load needed files from CSV.
embed_flag <- TRUE


#---- Initialization ----
## Load in documents. 
## This creates a "library" of executive order documents that
## will be referenced to summarize information.

if (!init_flag) {
  # To skip initialization, load the corpus from CSV.
  eo_corpus <- read_csv("eo_corpus.csv")
} else {
  # continue with initialization 
  
  # load the original meta data for reference
  fname <- paste0("documents_signed_in_2025_signed_by_",
                  "donald_trump_of_type_presidential_document_",
                  "and_of_presidential_document_type_executive_order-1.csv")
  eo_meta <- read_csv(fname)
  
  # Create a new data frame with the
  # executive_order_number, pub date, and title columns from eo_meta. 
  # Add an empty column to store the text retrieved from each document.
  eo_base <- eo_meta %>%
    filter(!is.na(pdf_url), !is.na(executive_order_number)) %>%
    select(executive_order_number, publication_date, title) %>%
    mutate(text = NA_character_)
  
  # Iterate through the executive order numbers in eo_base.
  # The number is used to concatenate a file name for each
  # of the text files generated in the scrape_eos script.
  # The text is loaded and stored in a new column.
  for (i in seq_len(nrow(eo_base))) {
    eo_number <- eo_base$executive_order_number[i]
    
    filename <- glue("executive_orders_cleaned/EO_{eo_number}.txt")
    
    if (!file.exists(filename)) {
      message(glue("File for EO {eo_number} does not exist."))
      next
    }
    text <- read_file(filename)
    eo_base$text[i] <- text
  }
  
  # Copy the data to eo_corpus so that we retain this base file.
  eo_corpus <- eo_base
  # Clean the text data:
  eo_corpus <- eo_corpus %>%
    rename(eo_id = executive_order_number) %>%  # Rename executive_order_number column
    mutate(clean_text = gsub("\\s+", " ", text),   # Remove excessive white space
           clean_text = trimws(clean_text)) %>%
    select(-text) %>%                          # Drop the text column
    mutate(clean_text = tolower(clean_text))   # Convert all text to lower case.
  
  ## eo_corpus is now a reference data set
  ## with full text of EOs. The clean_text
  # column includes that text.
  # Save to CSV for future use.
  write_csv(eo_corpus, "eo_corpus.csv")
}


#---- Chunk RAG Corpus ----

# Create a function that will chunk the text in eo_corpus
# based on sentences. Use a period to identify the end of a sentence.
chunk_text_by_sentence <- function(text, min_chunk_size = 5) {
  # Split the text into sentences based on periods.
  sentences <- unlist(strsplit(text, "(?<=\\.)\\s+", perl = TRUE))
  
  # Drop sentences that have fewer than min_chunk_size characters.
  # We assume these do not have significant value.
  sentence_lengths <- nchar(sentences)
  sentences <- sentences[sentence_lengths >= min_chunk_size]
  
  # Return the chunks (sentences) as a character vector.
  return(sentences)
}

# This is controlled by the embed_flag
if (!embed_flag) {
  # Load the already chunked data from CSV.
  chunked_corpus <- read_csv("chunked_corpus.csv")
} else {
  # continue with chunking eo_corpus
  
  chunked_corpus <- eo_corpus %>%
    # Apply the chunk_text_by_sentence function to 
    # each row in the clean_text column.
    mutate(chunks = map(clean_text, ~ chunk_text_by_sentence(.x,
                                      min_chunk_size = 12))) %>%
    
    # Unnest the list of chunks into separate rows.
    # Each chunk gets its own row tied to the original
    # document id.
    unnest_longer(chunks, values_to = "chunk_text") %>%
    select(eo_id, chunk_text)
  
  # Now that we have the sentences, we can 
  # remove punctuation from the chunked text.
  chunked_corpus <- chunked_corpus %>%
    mutate(chunk_text = gsub("[[:punct:]]", "", chunk_text))
  
  # Add a column with the number of characters in each chunk.
  chunked_corpus <- chunked_corpus %>%
    mutate(chunk_length = nchar(chunk_text))
  
  # By each eo_id, create a chunk ID for
  # each chunk within that document, and store in chunk_num.
  chunked_corpus <- chunked_corpus %>%
    group_by(eo_id) %>%
    mutate(chunk_id = row_number()) %>%
    ungroup() %>%
    select(eo_id, chunk_id, chunk_length, chunk_text)
  
  # write the chunked corpus to CSV for later use
  write_csv(chunked_corpus, "chunked_corpus.csv")
}

## When we exit this section, chunked_corpus has the
## chunked text in the chunk_text column.


#---- Embed RAG Corpus ----

## Need the Embedding model to embed query and chunks.
# Import the sentence_transformers Python library
sentence_transformers <- import("sentence_transformers")
# Load a specific pre-trained model: all-MiniLM-L6-v2
model <- sentence_transformers$SentenceTransformer("all-MiniLM-L6-v2")

if (!embed_flag) {
  # To skip embedding, load the embedded chunks from a file.
  embed_corpus <- readRDS("embed_corpus.rds")
} else {
  # continue with embedding
  
  # Copy the chunked data frame to preserve the original data
  embed_corpus <- chunked_corpus
  
  # Add new column to data frame with the embeddings.
  embed_corpus$embedding <- map(embed_corpus$chunk_text, 
                                function(text) model$encode(text))
  
  # The embed_corpus data frame now includes
  # the chunk_text and the associated embeddings.
  # Save to CSV for future use.
  saveRDS(embed_corpus, "embed_corpus.rds")
}

## When we exit this section, embed_corpus has the vector
## embeddings for each chunk in the embedding column.


#---- Fetch Top Chunks ----
## Functions to retrieve RAG chunks based on cosine similarity

## Define a function to return cosine similarity
cosine_sim <- function(a, b) {
  sum(a * b) / (sqrt(sum(a^2)) * sqrt(sum(b^2)))
}

# Fetch Top K chunks for query based on similarity scores.
fetch_top_k <- function(query_text, corpus_df, model, k = 3) {
  # encode the query based on the LLM model provided.
  query_embed <- model$encode(query_text)
  
  # Calculate similarity scores using our cosine_sim function.
  corpus_df <- corpus_df %>%
    mutate(similarity = map_dbl(embedding,
                                ~ cosine_sim(., query_embed))) %>%
    # sort the data frame by similarity scores
    arrange(desc(similarity)) %>%
    # return just the Top K results
    slice_head(n = k)
  
  # Return the data frame with the Top K results.
  return(corpus_df)
}


#---- Get Overview ----

# import the transformers library from Hugging Face
transformers <- import("transformers")

# Load an LLM model and prep it for generating text
generator <- transformers$pipeline("text-generation",
                                   model = "mistralai/Mistral-7B-Instruct-v0.1",
                                   device = 0)

# Function to return overview of EO content based on question submitted.
get_overview <- function(aquery, 
                       ragdf = embed_corpus, 
                       embed_model = model, 
                       top_k = 3,
                       LLMgenerator = generator) {
  
  # Get the top 3 results from the embedded corpus
  top_results <- fetch_top_k(aquery, ragdf, embed_model, top_k)
  
  context <- paste(top_results$chunk_text, collapse = "\n")
  
  # Format a prompt to send to the LLM
  myprompt <- glue(
    "<CONTEXT>\n{context}\n</CONTEXT>\n\n", 
    "<QUESTION>\n{aquery}\n</QUESTION>\n\n<ANSWER>"
  )

  # Get a response from the LLM generator
  response <- LLMgenerator(myprompt, max_new_tokens = 200L, do_sample=FALSE)
  output <- response[[1]]$generated_text
  # Use RegEx to strip out everything before the <ANSWER> tag
  output <- sub("(?s).*<ANSWER>\\s*", "", output, perl=TRUE)
  
  # Display the response
  return(output)
}

#---- Get Document Summaries ----
## Function to get summaries of the most relevant
## executive orders from the corpus based on the query.
get_doc_summaries <- function(aquery,
                              ragdf = embed_corpus,
                              full_corpus = eo_corpus,
                              embed_model = model,
                              top_k = 3,
                              LLMgenerator = generator) {
  
  # Get the Top K chunks
  top_chunks <- fetch_top_k(aquery, ragdf, embed_model, top_k)
  # Get the unique eo_id values from those chunks
  eo_ids <- unique(top_chunks$eo_id)
  
  # Get the full text for those eo_id values
  relevant_docs <- full_corpus %>%
    filter(eo_id %in% eo_ids)
  
  # Iterate through those documents and generate a summary for each one.
  summaries <- list()
  
  for (i in seq_len(nrow(relevant_docs))) {
    doc_text <- relevant_docs$clean_text[i]
    doc_id <- relevant_docs$eo_id[i]
    
    request <- glue(
      "Provide a summary of the document above in three bullet points. ",
      "You are analyzing executive orders signed by President Donald Trump in 2025. ",
      "Do not attribute them to any other president."
    )
    
    summary_prompt <- glue(
      "<DOCUMENT>\n{doc_text}\n</DOCUMENT>\n\n", 
      "<REQUEST>\n{request}\n</REQUEST><SUMMARY>"
    )
    
    # Call the LLM generator with the prompt
    summary_response <- LLMgenerator(summary_prompt, 
                                     max_new_tokens = 300L,
                                     do_sample=FALSE)
    # Ge the output text
    summary_output <- summary_response[[1]]$generated_text
    # Strip off everything before the <SUMMARY> tag
    summary_output <- sub("(?s).*<SUMMARY>\\s*", "", summary_output, perl=TRUE)
    # Replace the <SUMMARY> tag with the eo_id for clarity
    summary_output <- gsub("<SUMMARY>", paste0("<", doc_id, ">"), summary_output)
    summaries[[as.character(doc_id)]] <- summary_output
  }
  
  # Return the list of summaries
  return(summaries)
}

#---- Generate Full Response ----
# In one request, respond to a query with both
# an overview and document summaries.
generate_response <- function(query) {
  overview <- get_overview(query)
  summaries <- get_doc_summaries(query)
  
  answer <- paste0("Overview:\n", overview, "\n\nDocument Summaries:\n")
  for (eo_id in names(summaries)) {
    answer <- paste0(answer, "EO ID ", eo_id, "\n", summaries[[eo_id]], "\n\n")
  }
  answer <- gsub("</SUMMARY>", "", answer)
  return(answer)
}

#---- Test the Full Function ----

# Provide a query and test the full function
#my_question <- "What do the documents provided here say about gun control?"
#my_question <- "What did Trump say about immigration?"
#my_question <- "What do the documents provided here say regarding climate change?"
#my_question <- "What executive orders address healthcare reform?"
my_question <- "What are Trump's policies on education as outlined in these executive orders?"

the_response <- generate_response(my_question)
cat(the_response)
















