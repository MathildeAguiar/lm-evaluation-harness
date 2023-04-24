# TODO: Remove all TODO comments once the implementation is complete.
"""
TODO: Add the Paper Title on this line.
TODO: Add the paper's PDF URL (preferably from arXiv) on this line.

Word Sense Disambiguation (WSD) is a fundamental task for
NLP. The goal is determine among the different senses of a word
the most probable sense depending on the context. In this Task we evaluate the performance of 
generative pre-trained models on WSD. 

Homepage: TODO: Add the URL to the task's Homepage here.
"""
from lm_eval.base import Task, rf
from lm_eval.metrics import mean

# TODO: Add the BibTeX citation for the task.
_CITATION = """
"""


class VSD(Task):
    VERSION = 0
    DATASET_PATH = "GETALP/FLUE_VSD" 
    DATASET_NAME = None #"FSE"  

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True  

    # TODO delete, moved to _process_doc 
    def vec_to_sent(example):
        # Custom function to reconstruct the sentences into str and not arrays
        example['surface_forms'] = ' '.join(example['surface_forms']) 
        return example
    
    def split_dataset(self):
        wiki_split = self.dataset['wiki_FSE_verbs']
        train_valid = wiki_split.train_test_split(test_size=0.005, shuffle=True)  # TODO check the size it needs 
        train_split = train_valid['train']
        valid_split= train_valid['test']
        return train_split, valid_split

    def training_docs(self):
        if self.has_training_docs():
            # We cache training documents in `self._training_docs` for faster
            # few-shot processing. If the data is too large to fit in memory,
            # return the training data as a generator instead of a list.
            if self._training_docs is None:
                # TODO: Return the training document generator from `self.dataset`.
                # If you need to process the data, `map` over the documents with
                # the custom processing function, `self._process_doc`. E.g.
                # `map(self._process_doc, self.dataset["validation"])`
                # In most case you can leave this as is unless the dataset split is
                # named differently than the default `"train"`.
                train_split, valid_split = self.split_dataset()
                self._training_docs = list(map(self._process_doc, train_split)) 
                #list(self.dataset["train"])
            return self._training_docs 
          
    def validation_docs(self):
        if self.has_validation_docs():
            # TODO: Return the validation document generator from `self.dataset`.
            # If you need to process the data, `map` over the documents with the
            # custom processing function, `self._process_doc`. E.g.
            # `map(self._process_doc, self.dataset["validation"])`
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"validation"`.
            if self._validation_docs is None:
                train_split, valid_split = self.split_dataset()
                self._validation_docs = list(map(self._process_doc, valid_split))
            return self._validation_docs
    
    def test_docs(self):
        if self.has_test_docs():
            # TODO: Return the test document generator from `self.dataset`.
            # If you need to process the data, `map` over the documents with the
            # custom processing function, `self._process_doc`. E.g.
            # `map(self._process_doc, self.dataset["test"])`
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"test"`.
            #return self.dataset["test"]
            return map(self._process_doc, self.dataset["FSE"])
    
    
    def _process_doc(self, doc):
        # TODO: Process (detokenize, strip, replace etc.) each individual `doc`
        # with this function. You can map this across the docs in each available
        # dataset split. See the TODOs in `train_docs`, `validation_docs`, and
        # `test_docs` for snippets.
        # NOTE: DELETE THIS FUNCTION IF UNUSED.
        doc['surface_forms'] = ' '.join(doc['surface_forms']) 
        return doc

    # TODO refacto from here ##############################################
    """
    def doc_to_text(self, doc):
        # TODO: Format the query prompt portion of the document example.
        # TODO: find the right prompt to trigger the task 
        return (
            "Contexte: {}",
            "ID: {}"
            "\nQuestion: Est-ce que {} correspond au mot {} dans son contexte ?"
            "\nRéponse:".format(
                doc["surface_forms"],
                doc["instance_fine_pos"],  # TODO determine what should be put here, because there is no wn30_key
                doc["instance_surface_forms"]
                )
        )
    """
    
    # NOTE: Manual dummy 
    def doc_to_text(self, doc):
        return (
            "Contexte : Cela aboutit à la capture de Huntsville le 11 avril 1862 ."
            "\nID: d001.s012.t000"  
            "\nQuestion: Quel gloss correspond au verbe dont l'ID a été donné dans le contexte de cette phrase ?"
            "\nRéponse:"
        )

    def doc_to_target(self, doc):
        # TODO: Fill in the `target` ("gold answer") variable.
        # The prepended `" "` is required to space out the `doc_to_text` and
        # `doc_to_target` strings.
        #target = doc['disambiguate_labels']  #""
        #return " " + target
        #return " {}".format({0: "non", 1: "oui"}[doc["instance_fine_pos"]])  # TODO same as previous todo
        # TODO for the manual test
        return "  __ws_1_APlisé__adj__1"

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or
            test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        # TODO: Construct your language model requests with the request factory, `rf`,
        # and return them as an iterable.
        # TODO I have no idea how I should do it 
        #cont_request = rf.greedy_until(ctx, ["\nQuestion:"])
        #return cont_request
        #ll_yes, _ = rf.loglikelihood(ctx, " oui")
        #ll_no, _ = rf.loglikelihood(ctx, " non")
        #return ll_yes, ll_no
        ll, is_prediction = rf.loglikelihood(ctx, "  __ws_1_APlisé__adj__1")
        return is_prediction


    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        # TODO: For each (sub)metric in the task evaluation, add a key-value pair
        # with the metric name as key and the corresponding metric result as value
        # for the current `doc`.
        """       
        ll_yes, ll_no = results
        gold = doc["instance_fine_pos"]

        acc = 1.0 if (ll_yes > ll_no) == gold else 0.0

        return {"acc": acc}
        """
        (is_prediction,) = results
        return {"acc": is_prediction}

    def aggregation(self):
        """
        :returns: {str: [metric_score] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metric scores
        """
        # TODO: For each (sub)metric in the task evaluation, add a key-value pair
        # with the metric name as key and an aggregation function as value which
        # determines how to combine results from each document in the dataset.
        # Check `lm_eval.metrics` to find built-in aggregation functions.
        return {"acc": mean}

    def higher_is_better(self):
        # TODO: For each (sub)metric in the task evaluation, add a key-value pair
        # with the metric name as key and a `bool` value determining whether or
        # not higher values of that metric are deemed better.
        return {"acc": True}
