import dspy
from dspy.retrieve.chromadb_rm import ChromadbRM
from index_bridge_world_system import CHROMADB_DIR, CHROMA_COLLECTION_NAME


class ZeroShot(dspy.Module):
    """
    Provide answer to question
    """
    def __init__(self):
        super().__init__()
        self.prog = dspy.Predict("question -> answer")

    def forward(self, question):
        return self.prog(question="In the game of bridge, " + question)


class Definitions(dspy.Module):
    """
    Retrieve the definition from Wikipedia (2017 version)
    """
    def __init__(self):
        super().__init__()
        self.retriever = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

    def forward(self, term):
        return self.retriever(f"In the game of bridge, what does {term} mean?")


class Terms(dspy.Signature):
    """
    List of extracted entities
    """
    question = dspy.InputField()
    answer = dspy.OutputField(format=list, desc="terms")


class FindTerms(dspy.Module):
    """
    Extract bridge terms from a question
    """
    def __init__(self):
        super().__init__()
        self.entity_extractor = dspy.Predict(Terms)

    def forward(self, question):
        max_num_terms = max(1, len(question.split())//4)
        prompt = f"Identify up to {max_num_terms} terms in the following question that are jargon in the card game bridge"
        prediction = self.entity_extractor(
            question=f"{prompt}\n{question}"
        )
        return dspy.Prediction(answer=prediction.answer)


class BiddingSystem(dspy.Module):
    """
    Rules for bidding in bridge
    """
    def __init__(self):
        super().__init__()
        from chromadb.utils import embedding_functions
        default_ef = embedding_functions.DefaultEmbeddingFunction()
        self.prog = ChromadbRM(CHROMA_COLLECTION_NAME, CHROMADB_DIR, default_ef, k=3)

    def forward(self, question):
        return self.prog(question)


class BridgeBiddingAdvisor(dspy.Module):
    """
    Functions as the orchestrator. All questions are sent to this module.
    """
    def __init__(self):
        super().__init__()
        self.definitions = Definitions()
        self.find_terms = FindTerms()
        self.bidding_system = BiddingSystem()
        self.prog = dspy.ChainOfThought("definitions, bidding_system, question -> answer",
                                        n=3)

    def forward(self, question):
        terms = self.find_terms(question)
        definitions = [self.definitions(term) for term in terms]
        bidding_system = self.bidding_system(question)
        return self.prog(definitions=definitions,
                         bidding_system=bidding_system,
                         question="In the game of bridge, " + question)


if __name__ == '__main__':
    import dspy_init
    dspy_init.init_gemini_pro(temperature=0.0)
    # dspy_init.init_gpt35(temperature=0.0)

    def run(name: str, module: dspy.Module, queries: [str], shorten: bool = False):
        print(f"**{name}**")
        for query in queries:
            response = module(query)
            if shorten:
                if type(response) == list:
                    response = [ f"{r['long_text'][:25]} ... {len(r['long_text'])}" for r in response]
            print(response)

    questions = [
        "What is Stayman?",
        "Playing Stayman and Jacoby Transfers, how do you respond with 5-5 in the majors?",
        "How do you respond to Strong 1NT with 5-5 in the majors?"
    ]

    run("Zeroshot", ZeroShot(), questions)
    # run("definitions", Definitions(), ["Stayman", "Jacoby Transfers", "Strong 1NT", "majors"])
    # run("find_terms", FindTerms(), questions)
    # run("bidding_system", BiddingSystem(), questions, shorten=True)
    # run("bidding_advisor", BridgeBiddingAdvisor(), questions)
