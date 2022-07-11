from dataclasses import dataclass, asdict
from sacrebleu import sacrebleu, corpus_bleu, corpus_chrf



@dataclass
class EvaluationResult:
    bleu: float
    chrf: float

    def as_dict(self):
        return asdict(self)

    def __iter__(self):
        return iter(self.as_dict().items())

    def metrics_str(self):
        return '; '.join(map(lambda e: f"{e[0].upper()}={round(e[1], 3)}", self))

    def __str__(self):
        return f"Evaluation[{self.metrics_str()}]"

    def __repr__(self):
        return self.__str__()


def evaluate(predictions, references, lang=None) -> EvaluationResult:
    if lang == 'zh':
        tokenize = 'zh'
    elif lang == 'ja':
        tokenize = 'ja-mecab'
    else:
        tokenize = sacrebleu.DEFAULT_TOKENIZER

    bleu = corpus_bleu(predictions, [references], tokenize=tokenize, force=True)
    chrf = corpus_chrf(predictions, [references])
    return EvaluationResult(bleu.score, chrf.score)
