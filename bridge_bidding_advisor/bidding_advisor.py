import dspy
from dspy import teleprompt
from dspy.retrieve.chromadb_rm import ChromadbRM
from index_bridge_world_system import CHROMADB_DIR, CHROMA_COLLECTION_NAME
import json

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
        result = self.retriever(f"In the game of bridge, what does {term} mean?", k=1)
        if result:
            return result[0].long_text
        return ""


class FindTerms(dspy.Module):
    """
    Extract bridge terms from a question
    """
    def __init__(self):
        super().__init__()
        self.entity_extractor = dspy.Predict("question -> terms")

    def forward(self, question):
        max_num_terms = max(1, len(question.split())//4)
        prompt = f"Identify up to {max_num_terms} terms in the following question that are jargon in the card game bridge."
        prediction = self.entity_extractor(
            question=f"{prompt}\n{question}"
        )
        answer = prediction.terms
        if "Terms: " in answer:
            start = answer.rindex("Terms: ") + len("Terms: ")
            answer = answer[start:]
        return [a.strip() for a in answer.split(',')]


def BiddingSystem():
    """
    Retreives rules for bidding in bridge.
    This is just a retriever and does not have any language model.
    """
    from chromadb.utils import embedding_functions
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    return ChromadbRM(CHROMA_COLLECTION_NAME, CHROMADB_DIR, default_ef, k=3)


class AdvisorSignature(dspy.Signature):
    definitions = dspy.InputField(format=str)  # function to call on input to make it a string
    bidding_system = dspy.InputField(format=str) # function to call on input to make it a string
    question = dspy.InputField()
    answer = dspy.OutputField()
    
    
class BridgeBiddingAdvisor(dspy.Module):
    """
    Functions as the orchestrator. All questions are sent to this module.
    """
    def __init__(self):
        super().__init__()
        self.find_terms = FindTerms()
        self.definitions = Definitions()
        # self.bidding_system = BiddingSystem()
        self.prog = dspy.ChainOfThought(AdvisorSignature, n=3)

    def forward(self, question):
        print("a:", question)
        terms = self.find_terms(question)
        print("b:", terms)
        definitions = [self.definitions(term) for term in terms]
        print("c:", definitions)
        bidding_system = BiddingSystem()(question)
        print("d:", shorten_list(bidding_system))
        prediction = self.prog(definitions=definitions,
                               bidding_system=bidding_system,
                               question="In the game of bridge, " + question,
                               max_tokens=-1024)
        return prediction.answer
    

def shorten_list(response):
    if type(response) == list:
        return [ f"{r['long_text'][:25]} ... {len(r['long_text'])}" for r in response]
    else:
        return response

if __name__ == '__main__':
    import dspy_init
    dspy_init.init_gemini_pro(temperature=0.0)
    #dspy_init.init_gpt35(temperature=0.0)

    def run(name: str, module: dspy.Module, queries: [str], shorten: bool = False):
        print(f"**{name}**")
        for query in queries:
            response = module(query)
            if shorten:
                response = shorten_list(response)
            print(response)
        print()

    questions = [
        "What is Stayman?",
        "When do you use Jacoby Transfers?",
        "Playing Stayman and Transfers, what do you bid with 5-4 in the majors?"
    ]

    run("Zeroshot", ZeroShot(), questions)
    run("definitions", Definitions(), ["Stayman", "Jacoby Transfers", "Strong 1NT", "majors"])
    run("find_terms", FindTerms(), questions)
    run("bidding_system", BiddingSystem(), questions, shorten=True)
    run("bidding_advisor", BridgeBiddingAdvisor(), questions)
    # exit(0)
      
    # create labeled training dataset
    traindata = json.load(open("trainingdata.json", "r"))['examples']
    trainset = [dspy.Example(question=e['question'], answer=e['answer']) for e in traindata]
    
    # train
    teleprompter = teleprompt.LabeledFewShot()
    optimized_advisor = teleprompter.compile(student=BridgeBiddingAdvisor(), trainset=trainset)
    run("optimized", optimized_advisor, questions)
    