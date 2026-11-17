import time, requests
import xml.etree.ElementTree as ET


def get_articles(base_url: str, tool_name: str, email: str, query: str, batch_size: int) -> list[dict[str, object]]:
    all_articles = []
    #
    session = requests.Session()

    # fetch data from pubmed
    count, query_key, web_env = search_pubmed(
        session=session,
        query=query,
        tool_name=tool_name,
        base_url=base_url,
        email=email,
    )

    print("===== Fetching articles from PubMed =====")
    print(f"Found {count:,} papers\n")

    for i in range(0, count, batch_size):
        print(f"Processing articles: {i} - {i+batch_size}")
        #
        xml_data = make_request(
            session=session,
            base_url=base_url,
            endpoint="efetch",
            params={
                "db": "pubmed",
                "query_key": query_key,
                "WebEnv": web_env,
                "retstart": i,
                "retmax": batch_size,
                "retmode": "xml",
                "tool": tool_name,
                "email": email,
            }
        )
        #
        root = ET.fromstring(xml_data)
        #
        for my_article in root.findall("PubmedArticle"):
            my_record = parse_article(article=my_article)
            all_articles.append(my_record)

        # Respect NCBI request limits
        time.sleep(0.5)

    return all_articles


def make_request(base_url: str, session: requests.Session,  endpoint, params):

    response = session.get(f"{base_url}/{endpoint}.fcgi", params=params, timeout=60)

    response.raise_for_status()

    return response.content


def search_pubmed(session: requests.Session, query: str, base_url: str, tool_name: str, email: str):

    search_term = f"({query}) AND pubmed pmc open access[filter] AND hasabstract"

    xml_data = make_request(
        session=session,
        base_url=base_url,
        endpoint="esearch",
        params={
            "db": "pubmed",
            "term": search_term,
            "usehistory": "y",
            "retmax": 0,
            "tool": tool_name,
            "email": email,
        }
    )

    root = ET.fromstring(xml_data)

    count = int(root.findtext("Count", "0"))
    query_key = root.findtext("QueryKey")
    web_env = root.findtext("WebEnv")

    return count, query_key, web_env


def parse_article(article) -> dict:
    medline = article.find("MedlineCitation")
    pubmed_data = article.find("PubmedData")
    article_data = medline.find("Article")

    # PMID
    pmid = medline.findtext("PMID", "")

    # Title
    title_element = article_data.find("ArticleTitle")
    title = "".join(title_element.itertext()) if title_element is not None else ""

    # DOI
    doi = ""
    if pubmed_data is not None:
        for article_id in pubmed_data.findall(".//ArticleId"):
            if article_id.get("IdType") == "doi":
                doi = article_id.text or ""
                break
    # Fallback: ELocationID
    if not doi:
        for elocation in article_data.findall("ELocationID"):
            if elocation.get("EIdType") == "doi":
                doi = elocation.text or ""
                break

    # Authors
    authors = []
    for author in article_data.findall("./AuthorList/Author"):
        #
        collective = author.findtext("CollectiveName")
        #
        if collective:
            authors.append(collective)
            continue
        #
        last_name = author.findtext("LastName", "")
        fore_name = author.findtext("ForeName", "")
        #
        name = " ".join(part for part in [fore_name, last_name] if part)
        #
        if name:
            authors.append(name)

    # Abstract
    abstract_parts = []
    for element in article_data.findall("./Abstract/AbstractText"):
        text = "".join(element.itertext())
        label = element.get("Label")
        #
        if label:
            text = f"{label}: {text}"
        #
        abstract_parts.append(text)
    abstract = "\n".join(abstract_parts)

    # Author keywords
    keywords = []
    for keyword in medline.findall("./KeywordList/Keyword"):
        keyword_text = "".join(keyword.itertext())
        #
        if keyword_text:
            keywords.append(keyword_text)

    # Publication year
    year = article_data.findtext("./Journal/JournalIssue/PubDate/Year", "")
    #
    if not year:
        year = article_data.findtext("./Journal/JournalIssue/PubDate/MedlineDate", "")

    return {
        "pmid": pmid,
        "title": title,
        "doi": doi,
        "authors": " | ".join(authors),
        "abstract": abstract,
        "keywords": keywords,
        "year": year,
    }
