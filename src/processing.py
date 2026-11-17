from itertools import combinations


def process_articles(articles: list) -> tuple:
    # collect edge lists
    id_title_map = get_article_title_ids(articles)
    article_author_edge_list = get_article_author_edge_list(articles)
    author_author_edge_list = get_co_authors(articles)
    keyword_keyword_edge_list = get_keyword_keyword_edge_list(articles)
    author_keyword_edge_list = get_author_keyword_edge_list(articles)
    article_keyword_edge_list = get_article_keyword_edge_list(articles)

    #
    articles, authors, keywords = [], [], []
    #
    for my_article, my_author in article_author_edge_list:
        articles.append(my_article)
        authors.append(my_author)
    for my_author1, my_author2 in author_author_edge_list:
        authors.append(my_author1)
        authors.append(my_author2)
    for my_keyword1, my_keyword2 in keyword_keyword_edge_list:
        keywords.append(my_keyword1)
        keywords.append(my_keyword2)
    for my_author, my_keyword in author_keyword_edge_list:
        authors.append(my_author)
        keywords.append(my_keyword)
    for my_article, my_keyword in article_keyword_edge_list:
        articles.append(my_article)
        keywords.append(my_keyword)
    #
    articles = sorted(set(articles))
    authors = sorted(set(authors))
    keywords = sorted(set(keywords))

    data_dictionary = {
        "article": articles,
        "author": authors,
        "keyword": keywords,
        "id_article_map": id_title_map,
        ("article", "is_written_by", "author"): article_author_edge_list,
        ("article", "contains", "keyword"): article_keyword_edge_list,
        ("author", "writes_with", "author"): author_author_edge_list,
        ("author", "works_on", "keyword"): author_keyword_edge_list,
        ("keyword", "appears_with", "keyword"): keyword_keyword_edge_list,
    }
    #
    node_types = ["article", "author", "keyword"]
    edge_types = [
        ("article", "is_written_by", "author"),
        ("article", "contains", "keyword"),
        ("author", "writes_with", "author"),
        ("author", "works_on", "keyword"),
        ("keyword", "appears_with", "keyword"),
    ]

    return data_dictionary, node_types, edge_types


def get_article_author_edge_list(articles: list) -> list[tuple[str, str]]:
    #
    edge_list = []
    for my_article in articles:
        author_list = my_article["authors"].split("|")
        for my_author in author_list:
            edge_list.append((my_article["pmid"], my_author.strip()))

    return edge_list


def get_article_title_ids(articles: list) -> dict[str, str]:
    id_dict = {}
    for my_article in articles:
        id_dict[my_article["pmid"]] = my_article["title"]

    return id_dict


def get_co_authors(articles: list) -> list[tuple[str, str]]:
    edge_list = []
    for my_article in articles:
        author_list = my_article["authors"].split("|")
        author_pairs = list(combinations(author_list, 2))
        edge_list.extend(author_pairs)

    return edge_list


def get_keyword_keyword_edge_list(articles: list) -> list[tuple[str, str]]:
    edge_list = []
    for my_article in articles:
        keyword_pairs = list(combinations(my_article["keywords"], 2))
        edge_list.extend(keyword_pairs)

    return edge_list


def get_author_keyword_edge_list(articles: list) -> list[tuple[str, str]]:
    edge_list = []
    for my_article in articles:
        author_list = my_article["authors"].split("|")
        for my_author in author_list:
            for my_keyword in my_article["keywords"]:
                edge_list.append((my_author, my_keyword))

    return edge_list


def get_article_keyword_edge_list(articles: list) -> list[tuple[str, str]]:
    edge_list = []
    for my_article in articles:
        for my_keyword in my_article["keywords"]:
            edge_list.append((my_article["pmid"], my_keyword))

    return edge_list
