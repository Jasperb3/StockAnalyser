import shutil
import yfinance as yf
import os
import uuid
import threading
from pathlib import Path
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from nasdaq_data.nasdaq_grabber import nasdaq_grabber
from stock_analyser.utils.fmp_get_filings import get_most_recent_filing, save_filings
from stock_analyser.tools.tool_utils.news_sentiment_util import get_news, scrape_text
from stock_analyser.utils.constants import KNOWLEDGE_DIR, FILINGS_DIR, DB_DIR
from crewai_tools import QdrantVectorSearchTool
import time

class Setup:
    def __init__(self, state):
        self.state = state
        # Initialize Gemini client
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if self.gemini_api_key:
            # genai.configure(api_key=self.gemini_api_key)
            self.genai_client = genai.Client(api_key=self.gemini_api_key)
        else:
            print("⚠️ GEMINI_API_KEY not found in environment variables")
            self.genai_client = None
        
        # Initialize Qdrant client
        self.qdrant_url = os.environ.get("QDRANT_CLUSTER_URL")
        self.qdrant_api_key = os.environ.get("QDRANT_API_KEY")
        if self.qdrant_url and self.qdrant_api_key:
            self.qdrant_client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
        else:
            print("⚠️ QDRANT_URL or QDRANT_API_KEY not found in environment variables")
            self.qdrant_client = None
        
        # Collection name for Qdrant
        self.collection_name = f"stock_knowledge" if hasattr(self.state, 'company_ticker') else "default_knowledge"
        
        # Initialize QdrantVectorSearchTool to None (will be set up later if Qdrant setup is successful)
        self.qdrant_tool = None
        
        # Keep track of processed files to avoid duplicates
        self.processed_files = set()

    def process_new_markdown_files(self):
        """Find and process new markdown files in the knowledge directory."""
        print("Checking for new markdown files...")
        
        md_files = list(KNOWLEDGE_DIR.glob("*.md"))
        new_files = [f for f in md_files if str(f) not in self.processed_files]
        
        if not new_files:
            print("No new markdown files found.")
            return 0
            
        print(f"Found {len(new_files)} new markdown files to process.")
        
        for file_path in new_files:
            try:
                # Extract text from the file
                text_chunks = self.extract_text_from_markdown(single_file_path=file_path)
                if not text_chunks:
                    print(f"⚠️ No text chunks extracted from {file_path}")
                    continue
                    
                # Generate embeddings
                embeddings_with_text = self.generate_gemini_embeddings(text_chunks)
                if not embeddings_with_text:
                    print(f"⚠️ No embeddings generated for {file_path}")
                    continue
                    
                # Store in Qdrant
                success = self.store_in_qdrant(embeddings_with_text)
                if success:
                    print(f"Successfully added {file_path} to Qdrant ✅")
                    self.processed_files.add(str(file_path))
                else:
                    print(f"❌ Failed to add {file_path} to Qdrant")
                    
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
        
        return len(new_files)

    def custom_gemini_embedding_fn(self, text):
        """Generate embeddings for text using Gemini API."""
        try:
            if not self.genai_client:
                print("❌ Gemini client not initialized. Cannot generate embeddings.")
                return None
                
            result = self.genai_client.models.embed_content(
                model="text-embedding-004",
                contents=text
            )
            return result.embeddings
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            return None
    
    def check_knowledge_dir(self):
        print("Checking knowledge directory...")
        # Get the expected marker file name
        marker_file = KNOWLEDGE_DIR / f"{self.state.company_ticker}.txt"

        # Archive files if the marker file doesn't exist or if it's a different ticker
        for file in KNOWLEDGE_DIR.glob("*.txt"):
            if file != marker_file:
                archive_dir = KNOWLEDGE_DIR / f"{file.stem}_archive"
                print(f"Archiving existing files to {archive_dir}...")
                archive_dir.mkdir(parents=True, exist_ok=True)
                # Archive .md files
                for old_file in KNOWLEDGE_DIR.glob("*.md"):
                    new_file_path = archive_dir / old_file.name
                    old_file.rename(new_file_path)
                file.unlink()  # Remove the old marker file

                # Remove the db directory
                db = DB_DIR
                if db.exists():
                    shutil.rmtree(db)

                # Delete the Qdrant collection
                if self.qdrant_client:
                    try:
                        if self.qdrant_client.collection_exists(self.collection_name):
                            self.qdrant_client.delete_collection(self.collection_name)
                            print(f"Deleted Qdrant collection: {self.collection_name}")
                    except Exception as e:
                        print(f"❌ Error deleting Qdrant collection: {e}")

        # Create/overwrite the marker file for the current ticker
        marker_file.touch()

        print("Knowledge directory sorted ✅\n")


    def download_filings(self):
        download_required = False
        existing_files = list(FILINGS_DIR.iterdir())

        if existing_files:
            if not any(file.name.startswith(self.state.company_ticker) for file in existing_files):
                for file in existing_files:
                    file.unlink()
                download_required = True
        else:
            download_required = True
            print(f"No existing files found for {self.state.company_name} ({self.state.company_ticker})\n")

        filing_types = ["10-Q", "10-K", "6-K", "8-K"]
        success = True

        if download_required:
            print(f"Downloading SEC filings for {self.state.company_name} ({self.state.company_ticker})...")

            for filing_type in filing_types:
                try:
                    file_path = FILINGS_DIR / f"{self.state.company_ticker}_{filing_type}.txt"
                    try:
                        filings = get_most_recent_filing(self.state.company_ticker, filing_type)
                    except Exception as e:
                        print(f"❌ Error getting {filing_type} filing: {e}")
                        success = False
                        continue

                    if not filings:  # Check for empty content *before* saving
                        print(f"No {filing_type} filings found or empty for {self.state.company_ticker}")
                        success = False
                        continue

                    try:
                        save_filings(self.state.company_ticker, filing_type, file_path)
                        print(f"{filing_type} filing saved to {file_path} ✅")
                        success = True
                    except Exception as e:
                        print(f"❌ Error saving {filing_type} filing: {e}")
                        success = False
                        continue

                    if not file_path.stat().st_size:
                        print(f"❌ No content downloaded for {filing_type}")
                        success = False
                        continue  # Continue to the next filing type

                    # Conditional copy: only if it doesn't exist
                    knowledge_file_path = KNOWLEDGE_DIR / f"{self.state.company_ticker}_{filing_type}.md"
                    if not knowledge_file_path.exists():
                        try:
                            shutil.copyfile(str(file_path), str(knowledge_file_path))
                        except Exception as e:
                            print(f"❌ Error copying {filing_type} to knowledge dir: {e}")
                            success = False # Still track the failure
                            continue

                except Exception as e:  # Catch-all for unexpected errors in the loop
                    print(f"❌ Unexpected error processing {filing_type}: {e}")
                    success = False
                    continue

            if not success:
                print(f"❌ Failed to download required SEC filings for {self.state.company_name} ({self.state.company_ticker}).\nPlease verify the stock ticker and try again.")
            
            print("SEC filings downloaded successfully ✅\n")

        else:
            print(f"Using existing SEC filings for {self.state.company_name} ({self.state.company_ticker})")
            for filing_type in filing_types:
                file_path = FILINGS_DIR / f"{self.state.company_ticker}_{filing_type}.txt"
                knowledge_file_path = KNOWLEDGE_DIR / f"{self.state.company_ticker}_{filing_type}.md"
                if file_path.exists() and not knowledge_file_path.exists():
                    try:
                        shutil.copyfile(str(file_path), str(knowledge_file_path))
                    except Exception as e:
                        print(f"❌ Error copying existing {filing_type} to knowledge dir: {e}")

            print("SEC filings copied to the Knowledge directory ✅\n")

    
    def download_financial_data(self):
        print("Downloading financial data...")
        
        # get Nasdaq financial data
        grabber_instance = nasdaq_grabber()

        annual_financial_data = grabber_instance.nasdaq_financals(self.state.company_ticker, 1)
        annual_financial_data_path = f"{str(KNOWLEDGE_DIR)}/{self.state.company_ticker}_annual_financial_data.md"
        with open(annual_financial_data_path, "w") as f:
            f.write(annual_financial_data.to_markdown())

        semiannual_financial_data = grabber_instance.nasdaq_financals(self.state.company_ticker, 2)
        semiannual_financial_data_path = f"{str(KNOWLEDGE_DIR)}/{self.state.company_ticker}_semiannual_financial_data.md"
        with open(semiannual_financial_data_path, "w") as f:
            f.write(semiannual_financial_data.to_markdown())

        # get yfinance financial data
        company = yf.Ticker(self.state.company_ticker)

        historical_prices = company.history(period="5y", interval="1wk")
        historical_prices = historical_prices.to_dict(orient='index') if not historical_prices.empty else "❌ No historical prices found"

        historical_prices_path = f"{str(KNOWLEDGE_DIR)}/{self.state.company_ticker}_historical_prices.md"
        with open(historical_prices_path, "w") as f:
            f.write(f"# **{self.state.company_name} ({self.state.company_ticker}) 5y prices at 1wk intervals:**\n\n{historical_prices}\n\n")

        company_info = company.info
        financials = company.financials.to_dict(orient='index') if not company.financials.empty else "❌ No financials found"
        eps_trend = company.eps_trend.to_dict(orient='index') if not company.eps_trend.empty else "❌ No eps trend found"
        balance_sheet = company.balance_sheet.to_dict(orient='index') if not company.balance_sheet.empty else "❌ No balance sheet found"
        cashflow = company.cashflow.to_dict(orient='index') if not company.cashflow.empty else "❌ No cashflow found"
        major_holders = company.major_holders.to_dict(orient='index') if not company.major_holders.empty else "❌ No major holders found"
        quarterly_income_stmt = company.quarterly_income_stmt.to_dict(orient='index') if not company.quarterly_income_stmt.empty else "❌ No quarterly income statement found"
        dividends = company.dividends.to_dict() if not company.dividends.empty else "❌ No dividends found"
        splits = company.splits.to_dict() if not company.splits.empty else "❌ No splits found"
        calendar = company.calendar if company.calendar else "❌ No calendar found"
        upgrades_downgrades = company.upgrades_downgrades.to_dict(orient='index') if not company.upgrades_downgrades.empty else "❌ No upgrades/downgrades found"
        analyst_price_targets = company.analyst_price_targets if company.analyst_price_targets else "❌ No analyst price targets found"
        # news = company.news if company.news else "❌ No news found"
        news = get_news(self.state.company_ticker, number_of_articles=12)
        sustainability = company.sustainability.to_dict(orient='index') if not company.sustainability.empty else "❌ No sustainability data found"

        # write yfinance financial data to file

        financial_data_path = f"{str(KNOWLEDGE_DIR)}/{self.state.company_ticker}_financial_data.md"
        with open(financial_data_path, "w") as f:
            f.write(f"# **Company info:**\n{company_info}\n\n")
            f.write(f"# **Financials:**\n{financials}\n\n")
            f.write(f"# **EPS trend:**\n{eps_trend}\n\n")
            f.write(f"# **Balance sheet:**\n{balance_sheet}\n\n")
            f.write(f"# **Cashflow:**\n{cashflow}\n\n")
            f.write(f"# **Major holders:**\n{major_holders}\n\n")
            f.write(f"# **Quarterly income statement:**\n{quarterly_income_stmt}\n\n")
            f.write(f"# **Dividends:**\n{dividends}\n\n")
            f.write(f"# **Splits:**\n{splits}\n\n")
            f.write(f"# **Calendar:**\n{calendar}\n\n")
            f.write(f"# **Upgrades/downgrades:**\n{upgrades_downgrades}\n\n")
            f.write(f"# **Analyst price targets:**\n{analyst_price_targets}\n\n")
            f.write(f"# **News:**\n{news}\n\n")
            f.write(f"# **Sustainability:**\n{sustainability}\n\n")

        print("Financial data downloaded to the Knowledge directory ✅\n")

    def extract_text_from_markdown(self, single_file_path=None):
        """Extract text from markdown files in the knowledge directory."""
        print("Extracting text from markdown files...")

        text_chunks = []
        
        if single_file_path:
            md_files = [Path(single_file_path)]  # Process only the specified file
        else:
            md_files = list(KNOWLEDGE_DIR.glob("*.md")) # Process all md files
        
        if not md_files:
            print("⚠️ No markdown files found in knowledge directory")
            return []
        
        # Define chunk size and overlap for better semantic search
        chunk_size = 1000  # characters
        chunk_overlap = 200  # characters for context preservation
        
        for md_file in md_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Skip empty files
                    if not content.strip():
                        print(f"⚠️ Empty file: {md_file}")
                        continue
                    
                    # Process the content into chunks with overlap
                    # This is a simple implementation - for production, consider using
                    # a library like langchain.text_splitter
                    current_position = 0
                    content_length = len(content)
                    
                    while current_position < content_length:
                        # Calculate end position with overlap
                        end_position = min(current_position + chunk_size, content_length)
                        
                        # If not at the end, try to find a natural break point (newline, period, etc.)
                        if end_position < content_length:
                            # Look for natural break points within the last 100 characters of the chunk
                            look_back_range = min(100, chunk_size // 10)
                            natural_break_pos = content.rfind('\n\n', end_position - look_back_range, end_position)
                            
                            if natural_break_pos != -1:
                                end_position = natural_break_pos
                            else:
                                # Try to find a period or other sentence ending
                                for punct in ['. ', '! ', '? ', '\n']:
                                    natural_break_pos = content.rfind(punct, end_position - look_back_range, end_position)
                                    if natural_break_pos != -1:
                                        end_position = natural_break_pos + 1  # Include the punctuation
                                        break
                        
                        # Extract the chunk
                        chunk = content[current_position:end_position].strip()
                        
                        if chunk:  # Only add non-empty chunks
                            text_chunks.append({
                                "text": chunk,
                                "source": md_file.name
                            })
                        
                        # Move position for next chunk, with overlap
                        current_position = end_position - chunk_overlap if end_position < content_length else content_length
                
                # Mark this file as processed
                self.processed_files.add(str(md_file))
                
            except Exception as e:
                print(f"❌ Error reading {md_file}: {e}")
        
        print(f"Extracted {len(text_chunks)} text chunks from {len(md_files)} markdown files ✅")
        return text_chunks

    def generate_gemini_embeddings(self, text_chunks):
        """Generate embeddings for text chunks using Gemini API."""
        print("Generating Gemini embeddings...")
        
        if not self.genai_client:
            print("❌ Gemini client not initialized. Cannot generate embeddings.")
            return []
        
        embeddings_with_text = []
        failed_chunks = 0
        batch_size = 10  # Process in batches to avoid overwhelming the API
        requests_per_minute = 150
        delay_between_requests = 60.0 / requests_per_minute  # Calculate delay in seconds
        
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(text_chunks) + batch_size - 1)//batch_size}...", end="\r")
            
            for chunk in batch:
                try:
                    # Generate embedding using Gemini
                    result = self.genai_client.models.embed_content(
                        model="text-embedding-004",
                        contents=chunk["text"]
                    )

                    # Correctly access the embedding values
                    embedding_values = result.embeddings[0].values  # Access the first embedding and its values
                    
                    # Add embedding to the chunk data
                    embeddings_with_text.append({
                        "text": chunk["text"],
                        "source": chunk["source"],
                        "embedding": embedding_values
                    })
                    time.sleep(delay_between_requests) # delay
                except Exception as e:
                    failed_chunks += 1
                    print(f"❌ Error generating embedding for chunk from {chunk['source']}: {e}")
                    # Continue processing other chunks despite this error
        
        if failed_chunks > 0:
            print(f"⚠️ Failed to generate embeddings for {failed_chunks} chunks out of {len(text_chunks)}")
        
        print(f"Generated {len(embeddings_with_text)} embeddings ✅")
        return embeddings_with_text

    def store_in_qdrant(self, embeddings_with_text):
        """Store text and embeddings in Qdrant."""
        print(f"Storing embeddings in Qdrant collection '{self.collection_name}'...")
        
        if not self.qdrant_client:
            print("❌ Qdrant client not initialized. Cannot store embeddings.")
            return False
        
        try:
            # Check if collection exists and has correct vector size
            vector_size = 768  # Gemini text-embedding-004 has 768 dimensions
            
            if self.qdrant_client.collection_exists(self.collection_name):
                try:
                    collection_info = self.qdrant_client.get_collection(self.collection_name)
                    existing_vector_size = collection_info.config.params.vectors.size
                    
                    if existing_vector_size != vector_size:
                        print(f"Collection '{self.collection_name}' exists but with incorrect vector size ({existing_vector_size}). Recreating with size {vector_size}.")
                        self.qdrant_client.delete_collection(self.collection_name)
                        create_collection = True
                    else:
                        print(f"Collection '{self.collection_name}' already exists with correct vector size. Updating points.")
                        create_collection = False
                except Exception as e:
                    print(f"❌ Error checking collection info: {e}")
                    print(f"Recreating collection '{self.collection_name}'.")
                    self.qdrant_client.delete_collection(self.collection_name)
                    create_collection = True
            else:
                print(f"Collection '{self.collection_name}' does not exist. Creating it.")
                create_collection = True
            
            # Create collection if needed
            if create_collection:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                )
            
            # Prepare points for Qdrant
            points = []
            for item in embeddings_with_text:
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=item["embedding"],
                    payload={
                        "text": item["text"],
                        "source": item["source"]
                    }
                ))
            
            # Upload points to Qdrant in batches
            if points:
                batch_size = 100  # Adjust based on your needs
                for i in range(0, len(points), batch_size):
                    batch = points[i:i+batch_size]
                    print(f"Uploading batch {i//batch_size + 1}/{(len(points) + batch_size - 1)//batch_size} to Qdrant...")
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=batch
                    )
                print(f"Successfully stored {len(points)} points in Qdrant ✅")
                return True
            else:
                print("⚠️ No points to store in Qdrant")
                return False
                
        except Exception as e:
            print(f"❌ Error storing embeddings in Qdrant: {e}")
            print(f"Error details: {str(e)}")  # More detailed error information
            return False

    def setup_qdrant_vector_search(self):
        """Extract text, generate embeddings, and store in Qdrant."""
        print("Setting up Qdrant vector search...")
        
        # Extract text from markdown files
        text_chunks = self.extract_text_from_markdown()
        if not text_chunks:
            print("❌ No text chunks extracted. Cannot proceed with vector search setup.")
            return False
        
        # Generate embeddings
        embeddings_with_text = self.generate_gemini_embeddings(text_chunks)
        if not embeddings_with_text:
            print("❌ No embeddings generated. Cannot proceed with vector search setup.")
            return False
        
        # Store in Qdrant
        success = self.store_in_qdrant(embeddings_with_text)
        
        if success:
            print("Qdrant vector search setup completed successfully ✅")
            
            # Initialize QdrantVectorSearchTool
            if self.qdrant_url and self.qdrant_api_key:
                print("Initializing QdrantVectorSearchTool...")
                self.qdrant_tool = QdrantVectorSearchTool(
                    qdrant_url=self.qdrant_url,
                    qdrant_api_key=self.qdrant_api_key,
                    collection_name=self.collection_name,
                    limit=5,
                    score_threshold=0.35,
                    custom_embedding_fn=self.custom_gemini_embedding_fn
                )
                print("QdrantVectorSearchTool initialized ✅")
            else:
                print("❌ Cannot initialize QdrantVectorSearchTool: missing Qdrant URL or API key")
                
        else:
            print("❌ Failed to set up Qdrant vector search")
        
        return success