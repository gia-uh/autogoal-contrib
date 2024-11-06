import datetime
import textwrap
from pathlib import Path
from autogoal.utils import nice_repr

import black
import enlighten
import numpy as np
import torch
import torch.nn.functional as F
from autogoal_transformers._utils import (
    download_models_info,
    to_camel_case,
    DOWNLOAD_MODE,
)
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)

from autogoal.kb import (
    algorithm,
    AlgorithmBase,
    Label,
    Sentence,
    Seq,
    Supervised,
    VectorCategorical,
    VectorContinuous,
    Word,
    MatrixContinuousDense,
    GeneratedText,
    Prompt,
)

from autogoal.grammar import DiscreteValue, CategoricalValue, ContinuousValue
from autogoal_transformers._utils import TASK_ALIASES
from autogoal.utils._process import is_cuda_multiprocessing_enabled
import time
import warnings
from tqdm import tqdm


@nice_repr
class TransformersWrapper(AlgorithmBase):
    """
    Base wrapper for transformers algorithms from huggingface
    """

    @classmethod
    def is_upscalable(cls) -> bool:
        return False

    def __init__(self):
        self._mode = "train"
        self.device = torch.cuda.current_device() if torch.cuda.is_available() and is_cuda_multiprocessing_enabled() else torch.device("cpu")
        device_name = torch.cuda.get_device_name(self.device)

    def train(self):
        self._mode = "train"

    def eval(self):
        self._mode = "eval"

    def run(self, *args):
        if self._mode == "train":
            return self._train(*args)
        elif self._mode == "eval":
            return self._eval(*args)

        raise ValueError("Invalid mode: %s" % self._mode)

    def print(self, *args, **kwargs):
        if not self.verbose:
            return

        print(*args, **kwargs)

    @classmethod
    def check_files(cls):
        """
        Checks if the pretrained model and tokenizer files are available locally.

        Returns
        -------
        bool
            True if the files are available locally, False otherwise.
        """
        try:
            AutoModel.from_pretrained(cls.name, local_files_only=True)
            AutoTokenizer.from_pretrained(cls.name, local_files_only=True)
            return True
        except:
            return False

    @classmethod
    def download(cls):
        """
        Downloads the pretrained model and tokenizer.
        """
        AutoModel.from_pretrained(cls.name, trust_remote_code=True)
        AutoTokenizer.from_pretrained(cls.name)

    def init_model(self, model_name=None, transformer_model_cls = None, tokenizer_cls = None, transformer_cls_kwargs = None, tokenizer_cls_kwargs = None):
        if transformer_cls_kwargs is None:
            transformer_cls_kwargs = dict()
            
        if tokenizer_cls_kwargs is None:
            tokenizer_cls_kwargs = dict()
            
        if self.model is None:
            if not self.__class__.check_files():
                self.__class__.download()
            try:
                transformer_model_cls = AutoModel if transformer_model_cls == None else transformer_model_cls
                tokenizer_cls = AutoTokenizer if transformer_model_cls == None else tokenizer_cls
                
                transformer_cls_kwargs['local_files_only'] = True
                self.model = transformer_model_cls.from_pretrained(
                    model_name if not model_name is None else self.name, 
                    **transformer_cls_kwargs
                ).to(self.device)
                
                tokenizer_cls_kwargs['local_files_only'] = True
                self.tokenizer = tokenizer_cls.from_pretrained(
                    model_name if not model_name is None else self.name, 
                    **tokenizer_cls_kwargs
                )
                
            except OSError as e:
                raise TypeError(
                    f"{self.name} requires to run `autogoal contrib download transformers`."
                )
            except Exception as e:
                raise e

class WordEmbeddingTransformer(TransformersWrapper):
    def __init__(
        self,
        *,
        verbose=True,
    ):
        self.verbose = verbose
        self.model = None
        super().__init__()
        
    def run(self, input: Word) -> VectorContinuous:
        pass

class TextGenerationTransformer(TransformersWrapper):
    def __init__(
        self,
        *,
        verbose=True,
    ):
        self.verbose = verbose
        self.model = None
        super().__init__()
        
    def run(self, input: Prompt) -> GeneratedText:
        pass

class PretrainedWordEmbedding(TransformersWrapper):
    def __init__(
        self,
        word_embedding_model: algorithm(Word, VectorContinuous, include=["Transformer"]), # type: ignore
        pooling_model: CategoricalValue("mean", "max", "cls"),  # type: ignore
        batch_size=DiscreteValue(32, 1024),  # type: ignore
        *,
        verbose=True,
    ):
        self.verbose = verbose
        self.merge_mode = pooling_model
        self.batch_size = batch_size
        self.word_embedding_model = word_embedding_model
        self.model = None
        self.tokenizer = None
        super().__init__()

    def _merge(self, vectors):
        if not vectors.size(0):
            return np.zeros(vectors.size(1), dtype="float32")
        if self.merge_mode == "avg":
            return vectors.mean(dim=0).numpy()
        elif self.merge_mode == "first":
            return vectors[0, :].numpy()
        else:
            raise ValueError("Unknown merge mode")

    def run(self, input: Seq[Word]) -> MatrixContinuousDense:
        self.init_model(model_name=self.word_embedding_model.name)

        self.print("Tokenizing...", end="", flush=True)
        tokens = [self.tokenizer.tokenize(x) for x in input]
        sequence = self.tokenizer.encode_plus(
            [t for tokens in tokens for t in tokens],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        self.print("done")

        with torch.no_grad():
            self.print("Embedding...", end="", flush=True)
            output = self.model(**sequence).last_hidden_state
            output = output.squeeze(0)
            self.print("done")

        # delete the reference so we can clean the GRAM
        del sequence

        count = 0
        matrix = []
        for i, token in enumerate(input):
            contiguous = len(tokens[i])
            vectors = output[count : count + contiguous, :]
            vector = self._merge(vectors.to("cpu"))
            matrix.append(vector)
            count += contiguous

            # delete the reference so we can clean the GRAM
            del vectors
            torch.cuda.empty_cache()

        matrix = np.vstack(matrix)
        return matrix

class PretrainedSequenceEmbedding(TransformersWrapper):
    def __init__(
        self,
        word_embedding_model: algorithm(Word, VectorContinuous, include=["Transformer"]), # type: ignore
        batch_size: DiscreteValue(32, 1024),  # type: ignore
        pooling_strategy: CategoricalValue("mean", "max", "cls", "rms", "mean:max", "first:last"),  # type: ignore
        normalization_strategy: CategoricalValue("l2", "l1", "min-max", "z-score", "none"),  # type: ignore
        *,
        verbose=False,
    ):
        self.word_embedding_model = word_embedding_model
        self.model = None
        self.tokenizer = None
        self.batch_size = batch_size
        self.pooling_strategy = pooling_strategy
        self.normalization_strategy = normalization_strategy
        self.embedding_dim = None
        super().__init__()
        
    def init_model(self, transformer_model_cls = None, tokenizer_cls = None, transformer_cls_kwargs = {}, tokenizer_cls_kwargs = {}):
        super().init_model(self, self.word_embedding_model.name, transformer_model_cls, tokenizer_cls, transformer_cls_kwargs, tokenizer_cls_kwargs)
        try:
            self.embedding_dim = self.model.config.hidden_size * (self.pooling_strategy.count(":") + 1)
        except OSError as e:
            raise TypeError(
                f"{self.name} requires to run `autogoal contrib download transformers`."
            )
        except Exception as e:
            raise e 

    def run_unbatched(self, input) -> MatrixContinuousDense:
        self.init_model()
        
        with torch.no_grad():
            self.print("Embedding...", end="", flush=True)
            outputs = self.model(**input)
            hidden_states = outputs.last_hidden_state
            self.print("done")

            # Use the pooling strategy
            embeddings = self.pool(hidden_states)

            # Use the normalization strategy
            if self.normalization_strategy != "none":
                embeddings = self.normalize(embeddings)

        # Delete the reference to help with GPU memory management
        del outputs
        torch.cuda.empty_cache()

        return embeddings

    def run(self, input: Seq[Sentence]) -> MatrixContinuousDense:
        self.init_model()
        embeddings = []
        for i in tqdm(range(0, len(input), self.batch_size), desc="Processing batches"):
            batch_sentences = input[i : i + self.batch_size]
            self.print("Tokenizing...", end="", flush=True)
            encoded_input = self.tokenizer(
                batch_sentences, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            self.print("done")

            with torch.no_grad():
                self.print("Embedding...", end="", flush=True)
                outputs = self.model(**encoded_input)
                hidden_states = outputs.last_hidden_state

                self.print("done")

                # Use the pooling strategy
                batch_seq_embeddings = self.pool(hidden_states)

                # Use the normalization strategy
                if self.normalization_strategy != "none":
                    batch_seq_embeddings = self.normalize(batch_seq_embeddings)

                embeddings.extend(batch_seq_embeddings)

            # delete the reference so we can clean the GRAM
            del encoded_input, outputs
            torch.cuda.empty_cache()

        matrix = np.vstack(embeddings)
        return matrix

    def pool(self, hidden_states):
        if self.pooling_strategy == "mean":
            batch_seq_embeddings = hidden_states.mean(dim=1)
        elif self.pooling_strategy == "max":
            batch_seq_embeddings = hidden_states.max(dim=1).values
        elif self.pooling_strategy == "cls":
            batch_seq_embeddings = hidden_states[:, 0, :]
        elif self.pooling_strategy == "rms":
            batch_seq_embeddings = torch.sqrt((hidden_states**2).mean(dim=1))
        elif self.pooling_strategy == "mean:max":
            batch_seq_embeddings = torch.cat(
                [
                    hidden_states.mean(dim=1),
                    hidden_states.max(dim=1).values,
                ],
                dim=1,
            )
        elif self.pooling_strategy == "mean:max:rms":
            batch_seq_embeddings = torch.cat(
                [
                    hidden_states.mean(dim=1),
                    hidden_states.max(dim=1).values,
                    torch.sqrt((hidden_states**2).mean(dim=1)),
                ],
                dim=1,
            )
        elif self.pooling_strategy == "first:last":
            batch_seq_embeddings = torch.cat(
                (hidden_states[:, 0, :], hidden_states[:, -1, :]), dim=-1
            )
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        return batch_seq_embeddings.to("cpu")

    def normalize(self, embeddings):
        if self.normalization_strategy == "l2":
            normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        elif self.normalization_strategy == "l1":
            normalized_embeddings = F.normalize(embeddings, p=1, dim=1)
        elif self.normalization_strategy == "min-max":
            min_val = embeddings.min(dim=1, keepdim=True)[0]
            max_val = embeddings.max(dim=1, keepdim=True)[0]
            normalized_embeddings = (embeddings - min_val) / (max_val - min_val)
        elif self.normalization_strategy == "z-score":
            mean = embeddings.mean(dim=1, keepdim=True)
            std = embeddings.std(dim=1, keepdim=True)
            normalized_embeddings = (embeddings - mean) / std
        elif self.normalization_strategy == "none":
            normalized_embeddings = embeddings
        else:
            raise ValueError(
                f"Unknown normalization strategy: {self.normalization_strategy}"
            )
        return normalized_embeddings.to("cpu")

class PretrainedTextGeneration(TransformersWrapper):
    def __init__(
        self,
        batch_size=DiscreteValue(32, 1024),  # type: ignore
        max_gen_seq_length=DiscreteValue(16, 512),  # type: ignore
        temperature=ContinuousValue(0.1, 1.9),  # type: ignore
        *,
        verbose=False,
    ):
        self.device = torch.cuda.current_device() if torch.cuda.is_available() and is_cuda_multiprocessing_enabled() else torch.device("cpu")
        device_name = torch.cuda.get_device_name(self.device)
        self.verbose = verbose
        self.model = None
        self.tokenizer = None
        self.batch_size = batch_size
        self.max_gen_seq_length = max_gen_seq_length
        self.temperature = temperature

    @classmethod
    def check_files(cls):
        """
        Checks if the pretrained model and tokenizer files are available locally.

        Returns
        -------
        bool
            True if the files are available locally, False otherwise.
        """
        try:
            if cls.is_encoder_decoder:
                AutoModelForSeq2SeqLM.from_pretrained(cls.name, local_files_only=True)
            else:
                AutoModelForCausalLM.from_pretrained(cls.name, local_files_only=True)
            AutoTokenizer.from_pretrained(cls.name, local_files_only=True)
            return True
        except:
            return False

    @classmethod
    def download(cls):
        """
        Downloads the pretrained model and tokenizer.
        """
        if cls.is_encoder_decoder:
            AutoModelForSeq2SeqLM.from_pretrained(cls.name)
        else:
            AutoModelForCausalLM.from_pretrained(cls.name)

        AutoTokenizer.from_pretrained(cls.name)

    def init_model(self, transformer_model_cls = None, tokenizer_cls = None):
        if self.model is None:
            if not self.__class__.check_files():
                self.__class__.download()
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.name, local_files_only=True
                )

                if self.is_encoder_decoder:
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.name, local_files_only=True
                    ).to(self.device)
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.name, local_files_only=True
                    ).to(self.device)

                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token

                    # for decoder only architectures
                    self.tokenizer.padding_side = "left"

            except OSError as e:
                raise TypeError(
                    f"{self.name} requires to run `autogoal contrib download transformers`."
                )
            except Exception as e:
                raise e

    def generate(self, prompts):
        tokenized_batch = self.tokenizer.batch_encode_plus(
            prompts, return_tensors="pt", padding="longest", truncation=True
        ).to(self.device)

        generated_texts = []
        with torch.no_grad():
            self.print("Generating text...", end="", flush=True)
            input_ids = tokenized_batch["input_ids"]
            attention_mask = tokenized_batch["attention_mask"]
            max_length = max(self.max_gen_seq_length, input_ids.size()[1])

            output_sequences = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                temperature=self.temperature,
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            self.print("done")

            generated_texts = self.tokenizer.batch_decode(
                output_sequences, skip_special_tokens=True
            )
            # delete the reference so we can clean the GRAM
            del tokenized_batch

        torch.cuda.empty_cache()
        return generated_texts

    def run(self, input: Seq[Prompt]) -> Seq[GeneratedText]:
        self.init_model()

        generated_texts = []
        for i in tqdm(range(0, len(input), self.batch_size)):
            batch_input = input[i : i + self.batch_size]

            self.print("Tokenizing...", end="", flush=True)
            tokenized_batch = self.tokenizer.batch_encode_plus(
                batch_input, return_tensors="pt", padding="longest", truncation=True
            ).to(self.device)
            self.print("done")

            with torch.no_grad():
                self.print("Generating text...", end="", flush=True)
                input_ids = tokenized_batch["input_ids"]
                attention_mask = tokenized_batch["attention_mask"]
                max_length = max(self.max_gen_seq_length, input_ids.size()[1])

                output_sequences = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    # max_length=max_length,
                    # temperature=self.temperature,
                    repetition_penalty=1.5,
                    max_new_tokens=max_length,
                    # do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                self.print("done")

            generated_texts += self.tokenizer.batch_decode(
                output_sequences, skip_special_tokens=True
            )

            # delete the reference so we can clean the GRAM
            del tokenized_batch

        torch.cuda.empty_cache()
        return generated_texts

class PretrainedTextClassifier(TransformersWrapper):
    """
    A class used to represent a Pretrained Text Classifier which is a wrapper around the Transformers library.

    ...

    Attributes
    ----------
    device : torch.device
        a device instance where the model will be run.
    verbose : bool
        a boolean indicating whether to print verbose messages.
    model : transformers.PreTrainedModel
        the pretrained model.
    tokenizer : transformers.PreTrainedTokenizer
        the tokenizer corresponding to the pretrained model."""

    def __init__(self, verbose=True) -> None:
        super().__init__()
        self.verbose = verbose
        self.model = None
        self.tokenizer = None

    @classmethod
    def check_files(cls):
        """
        Checks if the pretrained model and tokenizer files are available locally.

        Returns
        -------
        bool
            True if the files are available locally, False otherwise.
        """
        try:
            AutoModelForSequenceClassification.from_pretrained(
                cls.name, local_files_only=True
            )
            AutoTokenizer.from_pretrained(cls.name, local_files_only=True)
            return True
        except:
            return False

    @classmethod
    def download(cls):
        """
        Downloads the pretrained model and tokenizer.
        """
        AutoModelForSequenceClassification.from_pretrained(cls.name)
        AutoTokenizer.from_pretrained(cls.name)

    def print(self, *args, **kwargs):
        if not self.verbose:
            return

        print(*args, **kwargs)

    def _check_input_compatibility(self, X, y):
        """
        Checks if the input labels are compatible with the pretrained model labels.

        Parameters
        ----------
        X :
            The input data.
        y :
            The input labels.

        Raises
        ------
        AssertionError
            If the number of unique labels in y is not equal to the number of classes in the pretrained model.
        KeyError
            If a label in y is not present in the pretrained model labels.
        """
        labels = set(y)

        assert (
            len(labels) != self.num_classes
        ), f"Input is not compatible. Expected labels are different from petrained labels for model '{self.name}'."

        for l in labels:
            if l not in self.id2label.values():
                raise KeyError(
                    f"Input is not compatible, label '{l}' is not present in pretrained data for model '{self.name}'"
                )

    def _train(self, X, y=None):
        if not self._check_input_compatibility(X, y):
            raise Exception(
                f"Input is not compatible with target pretrained model ({self.name})"
            )
        return y

    def _eval(
        self, X: Seq[Sentence], y: Supervised[VectorCategorical]
    ) -> VectorCategorical:
        self.init_model()
        self.print("Tokenizing...", end="", flush=True)

        encoded_input = self.tokenizer(
            X, padding=True, truncation=True, return_tensors="pt"
        )

        self.print("done")

        input_ids = encoded_input["input_ids"].to(self.device)
        attention_mask = encoded_input["attention_mask"].to(self.device)

        with torch.no_grad():
            self.print("Running Inference...", end="", flush=True)
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output.logits
            self.print("done")

        classification_vector = []
        for i in range(logits.shape[0]):
            logits_for_sequence_i = logits[i]
            predicted_class_id = logits_for_sequence_i.argmax().item()
            classification_vector.append(predicted_class_id)

        torch.cuda.empty_cache()
        return classification_vector

    def run(
        self, X: Seq[Sentence], y: Supervised[VectorCategorical]
    ) -> VectorCategorical:
        return TransformersWrapper.run(self, X, y)

class PretrainedZeroShotClassifier(TransformersWrapper):
    def __init__(self, batch_size, verbose=False) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None
        self.tokenizer = None
        self.candidate_labels = None

    @classmethod
    def check_files(cls):
        try:
            pipeline("zero-shot-classification", model=cls.name, local_files_only=True)
            return True
        except:
            return False

    @classmethod
    def download(cls):
        pipeline("zero-shot-classification", model=cls.name)

    def print(self, *args, **kwargs):
        if not self.verbose:
            return

        print(*args, **kwargs)

    def _train(self, X, y=None):
        # Store unique classes from y as candidate labels
        self.candidate_labels = list(set(y))
        return self._eval(X)

    def _eval(self, X: Seq[Sentence], *args) -> VectorCategorical:
        if self.model is None:
            if not self.__class__.check_files():
                self.__class__.download()

            try:
                self.model = pipeline(
                    "zero-shot-classification",
                    model=self.name,
                    local_files_only=True,
                    device=self.device,
                )
            except OSError:
                raise TypeError(
                    "'Huggingface Pretrained Models' require to run `autogoal contrib download transformers`."
                )

        classification_vector = []
        self.print(
            f"Running Inference with batch size {self.batch_size}...",
            end="",
            flush=True,
        )
        start = time.time()

        count = 0
        for i in tqdm(range(0, len(X), self.batch_size), desc="Processing batches"):
            batch = X[i : i + self.batch_size]
            self.print(f"Batch {count} with {len(batch)} items", end="", flush=True)
            count += 1

            warnings.filterwarnings("ignore", category=UserWarning)
            results = self.model(batch, candidate_labels=self.candidate_labels)
            warnings.filterwarnings("default", category=UserWarning)

            for result in results:
                best_score_index = result["scores"].index(max(result["scores"]))
                predicted_class_id = result["labels"][best_score_index]
                classification_vector.append(predicted_class_id)
            self.print(f"done batch", end="", flush=True)

        end = time.time()
        self.print(f"done inference in {end - start} seconds", end="", flush=True)

        torch.cuda.empty_cache()
        return classification_vector

    def run(
        self, X: Seq[Sentence], y: Supervised[VectorCategorical]
    ) -> VectorCategorical:
        return TransformersWrapper.run(self, X, y)

class PretrainedTokenClassifier(TransformersWrapper):
    def __init__(self, verbose=True) -> None:
        super().__init__()
        self.verbose = verbose
        self.model = None
        self.tokenizer = None

    @classmethod
    def check_files(cls):
        try:
            AutoModel.from_pretrained(cls.name, local_files_only=True)
            AutoTokenizer.from_pretrained(cls.name, local_files_only=True)
            return True
        except:
            return False

    @classmethod
    def download(cls):
        AutoModel.from_pretrained(cls.name)
        AutoTokenizer.from_pretrained(cls.name)

    def print(self, *args, **kwargs):
        if not self.verbose:
            return

        print(*args, **kwargs)

    def _check_input_compatibility(self, X, y):
        """
        Checks if the input labels are compatible with the pretrained model labels.

        Parameters
        ----------
        X :
            The input data.
        y :
            The input labels.

        Raises
        ------
        AssertionError
            If the number of unique labels in y is not equal to the number of classes in the pretrained model.
        KeyError
            If a label in y is not present in the pretrained model labels.
        """
        labels = set(y)

        assert (
            len(labels) != self.num_classes
        ), f"Input is not compatible. Expected labels are different from petrained labels for model '{self.name}'."

        for l in labels:
            if l not in self.id2label.values():
                raise KeyError(
                    f"Input is not compatible, label '{l}' is not present in pretrained data for model '{self.name}'"
                )

    def _train(self, X, y=None):
        self._check_input_compatibility(X, y)
        return y

    def _eval(self, X: Seq[Word], *args) -> Seq[Label]:
        if self.model is None:
            if not self.__class__.check_files():
                self.__class__.download()

            try:
                self.model = AutoModelForTokenClassification.from_pretrained(
                    self.name, local_files_only=True
                ).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.name, local_files_only=True
                )
                self.classifier = torch.nn.Linear(
                    self.model.config.hidden_size, self.num_classes
                ).to(self.device)

            except OSError:
                raise TypeError(
                    "'Huggingface Pretrained Models' require to run `autogoal contrib download transformers`."
                )

        self.print("Tokenizing...", end="", flush=True)

        # Tokenize and encode the sentences
        encoded_inputs = self.tokenizer(
            X,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Move the encoded inputs to the device
        sequence = encoded_inputs.to(self.device)
        word_ids = encoded_inputs.word_ids()

        # Get the model's predictions
        with torch.no_grad():
            outputs = self.model(**sequence)

        predictions = torch.argmax(outputs.logits, dim=2)
        token_predictions = [
            self.model.config.id2label[t.item()] for t in predictions[0]
        ]

        word_labels = [0] * len(X)

        for i in range(len(token_predictions)):
            word_index = word_ids[i]
            if word_index == None:
                continue

            word_labels[word_index] = token_predictions[i]

        assert len(X) == len(word_labels), "Output does not match input sequence shape"

        del sequence
        torch.cuda.empty_cache()
        return word_labels

    def run(self, X: Seq[Word], y: Supervised[Seq[Label]]) -> Seq[Label]:
        return TransformersWrapper.run(self, X, y)


def get_task_alias(task):
    for alias in TASK_ALIASES:
        if task in alias.value:
            return alias
    return None


TASK_TO_ALGORITHM_MARK = {
    TASK_ALIASES.TextClassification: "TEXT_CLASS_",
    TASK_ALIASES.WordEmbeddings: "WORD_EMB_",
    TASK_ALIASES.TextGeneration: "TEXT_GEN_",
    TASK_ALIASES.TokenClassification: "TOKEN_CLASS_",
}

TASK_TO_WRAPPER_NAME = {
    TASK_ALIASES.WordEmbeddings: WordEmbeddingTransformer.__name__,
    TASK_ALIASES.TextGeneration: TextGenerationTransformer.__name__,
    TASK_ALIASES.TextClassification: PretrainedTextClassifier.__name__,
    TASK_ALIASES.TokenClassification: PretrainedTokenClassifier.__name__,
}


def build_transformers_wrappers(
    max_amount=1000, min_likes=100, min_downloads=1000, download_mode=DOWNLOAD_MODE.HUB
):
    path = Path(__file__).parent / "_generated.py"

    with open(path, "w") as fp:
        fp.write(
            textwrap.dedent(
                f"""
            # AUTOGENERATED ON {datetime.datetime.now()}
            ## DO NOT MODIFY THIS FILE MANUALLY

            from numpy import inf, nan

            from autogoal.grammar import ContinuousValue, DiscreteValue, CategoricalValue, BooleanValue
            from autogoal_transformers._builder import {",".join([TASK_TO_WRAPPER_NAME[task] for task in TASK_TO_WRAPPER_NAME])}
            from autogoal.kb import *
            """
            )
        )

        for target_task in TASK_ALIASES:
            manager = enlighten.get_manager()

            imports = download_models_info(
                target_task,
                max_amount=max_amount,
                min_likes=min_likes,
                min_downloads=min_downloads,
                download_mode=download_mode,
            )

            counter = manager.counter(total=len(imports), unit="classes")

            for cls in imports:
                counter.update()
                _write_class(cls, fp, target_task)

            counter.close()
            manager.stop()

    black.reformat_one(
        path, True, black.WriteBack.YES, black.FileMode(), black.Report()
    )


def _write_class(item, fp, target_task):
    class_name = TASK_TO_ALGORITHM_MARK[target_task] + to_camel_case(
        item["name"].replace("-", "_")
    )
    print("Generating class: %r" % class_name)

    task = get_task_alias(item["metadata"]["task"])
    target_task = task if task is not None else target_task

    base_class = TASK_TO_WRAPPER_NAME[target_task]

    class_metadata = f"""
    class {class_name}({base_class}):
        name = "{item['name']}"
        id2label = {item['metadata']['id2label']}
        num_classes = {len(item['metadata']['id2label'])}
        tags = {len(item['metadata']['id2label'])}
        model_type = "{item['metadata']['model_type']}"
        architectures = {item['metadata']['architectures']}
        vocab_size = {item['metadata']['vocab_size']}
        type_vocab_size = {item['metadata']['type_vocab_size']}
        is_decoder = {item['metadata']['is_decoder']}
        is_encoder_decoder = {item['metadata']['is_encoder_decoder']}
        num_layers = {item['metadata']['num_layers']}
        hidden_size = {item['metadata']['hidden_size']}
        num_attention_heads = {item['metadata']['num_attention_heads']}
    """

    class_init = f"""
        def __init__(
            self, 
        ):
            {base_class}.__init__(
                self, 
            )
        """

    class_definition = textwrap.dedent(class_metadata + class_init)
    fp.write(class_definition)

    print("Successfully generated" + class_name)
    fp.flush()


                # batch_size=batch_size, 
                # max_gen_seq_length=max_gen_seq_length,
                # temperature=temperature

if __name__ == "__main__":
    build_transformers_wrappers(
        max_amount=100,
        download_mode=DOWNLOAD_MODE.BASE,
    )
