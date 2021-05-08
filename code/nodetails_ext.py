from nodetails.extractive import *

if __name__ == "__main__":
    extsum = get_summary_from_url("https://en.wikipedia.org/wiki/Citation_needed")

    print("\n == SENTENCES ==\n")
    for it in extsum.reference:
        print(it, extsum.sentences[it])


    print("\n  == SUMMARY ==\n")
    print(extsum.summary)

# END OF nodetails_ext.py
