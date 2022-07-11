from typing import Dict, Optional, TYPE_CHECKING

from data.encoder.sense import SenseSentenceBatchEncoder

if TYPE_CHECKING:
    from data.entities import AnnotatedSpan, WSDMTItem


def load_synset_clustering() -> Dict[str, str]:
    mapping = {}

    for line in open('data/clusters.tsv'):
        cluster_id, *synsets = line.rstrip().split('\t')
        mapping[cluster_id] = cluster_id
        for synset in synsets:
            mapping[synset] = cluster_id

    return mapping


class ClusteredSenseSentenceBatchEncoder(SenseSentenceBatchEncoder):
    _CLUSTERING = load_synset_clustering()

    def choose_span_label(self, span: 'AnnotatedSpan', item: Optional['WSDMTItem'] = None) -> Optional[str]:
        if not self.use_span(span):
            return None

        return ClusteredSenseSentenceBatchEncoder._CLUSTERING[span.sense]
