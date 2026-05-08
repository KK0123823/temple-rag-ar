from backend.query_chroma import query

def main():
    q = "文昌帝君有哪些禁忌？"
    res = query(q, top_k=4, entity="文昌帝君", section="禁忌")
    print(f"Query: {q}")
    print("-" * 60)
    for i, r in enumerate(res, start=1):
        print(f"rank: {i} distance: {r['distance']}")
        print(f"meta: {r['meta']}")
        print(f"doc: {r['doc'][:200]}...")
        print("-" * 60)

if __name__ == "__main__":
    main()
