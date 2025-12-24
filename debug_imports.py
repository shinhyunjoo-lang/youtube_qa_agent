
import pkgutil
import langchain
import langchain_community

print(f"Langchain path: {langchain.__path__}")
print(f"Langchain Community path: {langchain_community.__path__}")

try:
    from langchain.chains.summarize import load_summarize_chain
    print("Found in langchain.chains.summarize")
except ImportError:
    print("Not found in langchain.chains.summarize")

try:
    from langchain.chains import load_summarize_chain
    print("Found in langchain.chains")
except ImportError:
    print("Not found in langchain.chains")

