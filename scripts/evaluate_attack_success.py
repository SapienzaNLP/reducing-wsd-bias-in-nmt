import re
import os
import sys
import json
import spacy
import string
import logging
import argparse

from nltk.corpus import stopwords


def _process_strings(line,
                     lang_nlp,
                     get_lemmas,
                     get_pos,
                     remove_stopwords,
                     replace_stopwords,
                     get_maps):

    """ Helper function for obtaining various word representations """

    # strip, replace special tokens
    orig_line = line
    line = line.strip()
    line = re.sub(r'&apos;', '\'', line.strip())
    line = re.sub(r'&quot;', '\"', line.strip())
    # Tokenize etc.
    line_nlp = [t for t in lang_nlp(line) if t.pos_ != 'SPACE']
    spacy_tokens = [elem.text for elem in line_nlp]
    spacy_tokens_lower = [elem.text.lower() for elem in line_nlp]
    spacy_lemmas = None
    spacy_pos = None
    if get_lemmas:
        spacy_lemmas = list()
        for elem in line_nlp:
            if elem.lemma_ == '-PRON-' or elem.lemma_.isdigit():
                spacy_lemmas.append(elem.lower_)
            else:
                spacy_lemmas.append(elem.lemma_.lower().strip())
    if get_pos:
        spacy_pos = [elem.pos_ for elem in line_nlp]

    # Generate a mapping between whitespace tokens and SpaCy tokens
    ws_tokens = orig_line.strip().split()
    ws_tokens_lower = orig_line.strip().lower().split()
    ws_to_spacy_map = dict()
    spacy_to_ws_map = dict()
    if get_maps:
        ws_loc = 0
        ws_tok = ws_tokens[ws_loc]

        for spacy_loc, spacy_tok in enumerate(spacy_tokens):
            while True:
                # Map whitespace tokens to be identical to spacy tokens
                ws_tok = re.sub(r'&apos;', '\'', ws_tok)
                ws_tok = re.sub(r'&quot;', '\"', ws_tok)

                if spacy_tok == ws_tok or spacy_tok in ws_tok:
                    # Terminate
                    if ws_loc >= len(ws_tokens):
                        break

                    # Extend maps
                    if not ws_to_spacy_map.get(ws_loc, None):
                        ws_to_spacy_map[ws_loc] = list()
                    ws_to_spacy_map[ws_loc].append(spacy_loc)
                    if not spacy_to_ws_map.get(spacy_loc, None):
                        spacy_to_ws_map[spacy_loc] = list()
                    spacy_to_ws_map[spacy_loc].append(ws_loc)

                    # Move pointer
                    if spacy_tok == ws_tok:
                        ws_loc += 1
                        if ws_loc < len(ws_tokens):
                            ws_tok = ws_tokens[ws_loc]
                    else:
                        ws_tok = ws_tok[len(spacy_tok):]
                    break
                else:
                    ws_loc += 1

        # Assert full coverage of whitespace and SpaCy token sequences by the mapping
        ws_covered = sorted(list(ws_to_spacy_map.keys()))
        spacy_covered = sorted(list(set(list([val for val_list in ws_to_spacy_map.values() for val in val_list]))))
        assert ws_covered == [n for n in range(len(ws_tokens))], \
            'WS-SpaCy mapping does not cover all whitespace tokens: {}; number of tokens: {}'\
            .format(ws_covered, len(ws_tokens))
        assert spacy_covered == [n for n in range(len(spacy_tokens))], \
            'WS-SpaCy mapping does not cover all SpaCy tokens: {}; number of tokens: {}' \
            .format(spacy_covered, len(spacy_tokens))

    if remove_stopwords:
        # Filter out stopwords
        nsw_spacy_tokens_lower = list()
        nsw_spacy_lemmas = list()
        for tok_id, tok in enumerate(spacy_tokens_lower):
            if tok not in STOP_WORDS:
                nsw_spacy_tokens_lower.append(tok)
                if get_lemmas:
                    nsw_spacy_lemmas.append(spacy_lemmas[tok_id])
            else:
                if replace_stopwords:
                    nsw_spacy_tokens_lower.append('<STPWRD>')
                    if get_lemmas:
                        nsw_spacy_lemmas.append('<STPWRD>')

        spacy_tokens_lower = nsw_spacy_tokens_lower
        if get_lemmas:
            spacy_lemmas = nsw_spacy_lemmas

    return line_nlp, spacy_tokens_lower, spacy_lemmas, spacy_pos, ws_tokens, ws_tokens_lower, ws_to_spacy_map, \
        spacy_to_ws_map


def _get_nmt_label(translation,
                   adv_src,
                   true_src,
                   ambiguous_token,
                   ambiguous_token_loc_seed,
                   ambiguous_token_loc_adv,
                   attractor_tokens_loc,
                   seed_cluster_id,
                   adv_cluster_id,
                   other_cluster_ids,
                   sense_lemmas_to_cluster,
                   cluster_to_sense_lemmas,
                   sense_tokens_to_cluster,
                   cluster_to_sense_tokens,
                   alignments,
                   is_adv,
                   true_translation=None,
                   true_alignments=None):

    """ Helper function for evaluating whether the translations of true and adversarially perturbed source samples
    perform lexical WSD correctly. """

    # Lemmatize the translation for higher coverage of attack successes (preserve punctuation)
    _, spacy_tokens_lower, spacy_lemmas, _, ws_tokens, ws_tokens_lower, ws_to_spacy_map, spacy_to_ws_map = \
        _process_strings(translation,
                         tgt_nlp,
                         get_lemmas=True,
                         get_pos=False,
                         remove_stopwords=False,
                         replace_stopwords=False,
                         get_maps=True)

    # All sequences are Moses-tokenized and can be split at whitespace
    adv_src_tokens = adv_src.strip().split()
    true_src_tokens = true_src.strip().split()
    ambiguous_token_loc_seed = ambiguous_token_loc_seed[0]

    # Check that the provided locations of the ambiguous token are correct
    ambiguous_token_in_seed = true_src_tokens[ambiguous_token_loc_seed].lower().strip(punctuation_plus_space)

    assert ambiguous_token.lower().strip(punctuation_plus_space) == \
        ambiguous_token_in_seed or ambiguous_token[:-1] in ambiguous_token_in_seed, \
        'Ambiguous token \'{:s}\' does not match the true source token \'{:s}\' at the token location'\
        .format(ambiguous_token, ambiguous_token_in_seed)
    ambiguous_token_in_adv = adv_src_tokens[ambiguous_token_loc_adv].lower().strip(punctuation_plus_space)
    assert ambiguous_token.lower().strip(punctuation_plus_space) == \
        ambiguous_token_in_adv or ambiguous_token[:-1] in ambiguous_token_in_adv, \
        'Ambiguous token \'{:s}\' does not match the adversarial source token \'{:s}\' at the token location' \
        .format(ambiguous_token, ambiguous_token_in_adv)

    ambiguous_token_loc = ambiguous_token_loc_adv if is_adv else ambiguous_token_loc_seed

    other_cluster_lemmas = list()
    for cluster_id in other_cluster_ids:
        other_cluster_lemmas += cluster_to_sense_lemmas[cluster_id]
    other_cluster_tokens = list()
    for cluster_id in other_cluster_ids:
        other_cluster_tokens += cluster_to_sense_tokens[cluster_id]

    target_hits = list()
    attractor_term_translated = False
    # If alignments are available, look-up target sense aligned with the ambiguous source term
    alignments = alignments.strip()
    # Convert alignments into a more practical format, first
    line_align_table = dict()
    for word_pair in alignments.strip().split():
        src_id, tgt_id = word_pair.split('-')
        src_id = int(src_id)
        tgt_id = int(tgt_id)
        if not line_align_table.get(src_id, None):
            line_align_table[src_id] = [tgt_id]
        else:
            line_align_table[src_id].append(tgt_id)

    if ambiguous_token_loc in line_align_table.keys():
        homograph_aligned = True
        # Extend alignment window
        line_align_table_entry = sorted(line_align_table[ambiguous_token_loc])
        min_tgt_idx = max(0, line_align_table_entry[0])
        max_tgt_idx = min(len(spacy_lemmas) - 1, line_align_table_entry[-1])
        # Map from target whitespace to target spacy tokens
        min_tgt_idx = ws_to_spacy_map[min_tgt_idx][0]
        max_tgt_idx = ws_to_spacy_map[max_tgt_idx][0]
        tgt_window = range(min_tgt_idx, max_tgt_idx)
        # Check if aligned translation lemmas include correct / flipped source term translations
        aligned_translation_lemmas = [spacy_lemmas[idx] for idx in tgt_window]
        aligned_tgt_tokens = [spacy_tokens_lower[idx] for idx in tgt_window]

        for atl_id, atl in enumerate(aligned_translation_lemmas):
            # Skip stopwords
            if atl == '<STPWRD>':
                continue
            # Strip leading and trailing punctuation
            atl = atl.strip(punctuation_plus_space)
            # Match lemmas
            if sense_lemmas_to_cluster is not None:
                if sense_lemmas_to_cluster.get(atl, None):
                    target_hits.append(sense_lemmas_to_cluster[atl])
                else:
                    maybe_lemma_matches = sorted(cluster_to_sense_lemmas[seed_cluster_id] +
                                                 cluster_to_sense_lemmas[adv_cluster_id] +
                                                 other_cluster_lemmas, reverse=True, key=lambda x: len(x))

                    for maybe_lemma_match in maybe_lemma_matches:
                        if atl.startswith(maybe_lemma_match) or atl.endswith(maybe_lemma_match) or \
                                atl[:-1].endswith(maybe_lemma_match):
                            target_hits.append(sense_lemmas_to_cluster[maybe_lemma_match])
                            break
            # Match tokens
            if len(target_hits) == 0 and sense_tokens_to_cluster is not None:
                att = aligned_tgt_tokens[atl_id].strip(punctuation_plus_space)
                if sense_tokens_to_cluster.get(att, None):
                    target_hits.append(sense_tokens_to_cluster[att])
                else:
                    maybe_token_matches = \
                        sorted(cluster_to_sense_tokens[seed_cluster_id] +
                               cluster_to_sense_tokens[adv_cluster_id] +
                               other_cluster_tokens, reverse=True, key=lambda x: len(x))

                    for maybe_token_match in maybe_token_matches:
                        if att.startswith(maybe_token_match) or att.endswith(maybe_token_match.lower()) or \
                                att[:-1].endswith(maybe_token_match.lower()):
                            target_hits.append(sense_tokens_to_cluster[maybe_token_match.lower()])
                            break
    else:
        homograph_aligned = False

    # Check if the attractor term(s) have been translated (assumes translation if alignment was found)
    is_compound = False
    if is_adv:
        if line_align_table.get(attractor_tokens_loc[0], None) is not None:
            attractor_term_translated = True
        else:
            # Check if true and adversarial translations are identical
            if translation.strip() == true_translation.strip():
                attractor_term_translated = False
            # Try to check whether the attractor has been translated as part of a compound
            else:
                # Look up alignments for the seed translation
                true_line_align_table = dict()
                for word_pair in true_alignments.strip().split():
                    src_id, tgt_id = word_pair.split('-')
                    src_id = int(src_id)
                    tgt_id = int(tgt_id)
                    if not true_line_align_table.get(src_id, None):
                        true_line_align_table[src_id] = [tgt_id]
                    else:
                        true_line_align_table[src_id].append(tgt_id)

                # Check if the modified noun is aligned to the same position in both translations
                modified_align_true = true_line_align_table.get(attractor_tokens_loc[0], [])
                modified_align_adv = line_align_table.get(attractor_tokens_loc[0] + 1, [])
                true_translation_tokens = true_translation.strip().lower().split()
                aligned_true_tokens = [true_translation_tokens[true_loc].strip(punctuation_plus_space)
                                       for true_loc in modified_align_true]
                aligned_adv_tokens = [ws_tokens_lower[adv_loc].strip(punctuation_plus_space)
                                      for adv_loc in modified_align_adv]
                aligned_token_overlap = set(aligned_true_tokens) & set(aligned_adv_tokens)

                for true_token in aligned_true_tokens:
                    for adv_token in aligned_adv_tokens:
                        if true_token in aligned_token_overlap or adv_token in aligned_token_overlap:
                            continue
                        else:
                            if true_token != adv_token:
                                if true_token in adv_token or len(adv_token) > len(true_token) + 3 or \
                                        (adv_token not in true_token and true_token[-3:] == adv_token[-3:]):
                                    is_compound = True
                                    break
                if is_compound:
                    # Assume attractor and modified term have been jointly translated into a target compound
                    attractor_term_translated = True

    # If no alignments are available, match the known target sense
    if len(target_hits) == 0:
        for stl_id, stl in enumerate(spacy_lemmas):
            # if stl_id not in range(ambiguous_token_spacy_loc_seed - 3, ambiguous_token_spacy_loc_seed + 4):
            #     continue
            # Skip stopwords
            if stl == '<STPWRD>':
                continue
            # Strip leading and trailing punctuation
            stl = stl.strip(punctuation_plus_space)
            # Match lemmas
            if sense_lemmas_to_cluster is not None:
                if sense_lemmas_to_cluster.get(stl, None):
                    target_hits.append(sense_lemmas_to_cluster[stl])
                else:
                    maybe_lemma_matches = \
                        sorted(cluster_to_sense_lemmas[seed_cluster_id] +
                               cluster_to_sense_lemmas[adv_cluster_id] +
                               other_cluster_lemmas, reverse=True, key=lambda x: len(x))
                    for maybe_lemma_match in maybe_lemma_matches:
                        if stl.startswith(maybe_lemma_match) or stl.endswith(maybe_lemma_match) or \
                                stl[:-1].endswith(maybe_lemma_match):
                            target_hits.append(sense_lemmas_to_cluster[maybe_lemma_match])
                            break
            # Match tokens
            if len(target_hits) == 0 and sense_tokens_to_cluster is not None:
                stt = spacy_tokens_lower[stl_id].strip(punctuation_plus_space)
                if sense_tokens_to_cluster.get(stt, None):
                    target_hits.append(sense_tokens_to_cluster[stt])
                else:
                    maybe_token_matches = \
                        sorted(cluster_to_sense_tokens[seed_cluster_id] +
                               cluster_to_sense_tokens[adv_cluster_id] +
                               other_cluster_tokens, reverse=True, key=lambda x: len(x))
                    for maybe_token_match in maybe_token_matches:
                        if stt.startswith(maybe_token_match) or stt.endswith(maybe_token_match.lower()) or \
                                stt[:-1].endswith(maybe_token_match.lower()):
                            try:
                                target_hits.append(sense_tokens_to_cluster[maybe_token_match.lower()])
                                break
                            except KeyError:
                                pass

    # Source homograph is assumed to be translated if:
    # 1. Homograph is aligned
    # 2. Homograph is not aligned, but len(target_hits) > 0
    # 3. Homograph is not aligned and len(target_hits) == 0,
    #    but attractor modifies homograph and is translated into a compound
    if homograph_aligned:
        homograph_translated = True
    else:
        if len(target_hits) > 0:
            homograph_translated = True
        else:
            if is_adv:
                if is_compound and ambiguous_token_loc == (attractor_tokens_loc[0] + 1):
                    homograph_translated = True
                else:
                    homograph_translated = False
            else:
                homograph_translated = False

    # Flatten target hits
    target_hits = [hit[1] for hit_list in target_hits for hit in hit_list]
    # If target term is ambiguous, assume the translation is correct
    if seed_cluster_id in target_hits:
        return 'not_flipped', target_hits, attractor_term_translated

    elif adv_cluster_id in target_hits:
        return 'flipped_to_attr', target_hits, attractor_term_translated

    elif len(set(other_cluster_ids) & set(target_hits)) >= 1:
        return 'flipped_to_other', target_hits, attractor_term_translated

    # i.e. target_hits is empty
    else:
        if homograph_translated:
            return 'maybe_flipped', target_hits, attractor_term_translated

        else:
            return 'deleted_homograph', target_hits, attractor_term_translated


def _build_cluster_lookup(sense_clusters_table):

    """ Post-processes the scraped target sense cluster table by constructing a sense-to-cluster_id lookup table """

    # Initialize empty tables
    logging.info('Constructing the cluster lookup table ...')
    sense_to_cluster_table = dict()

    # Fill tables
    for src_term in sense_clusters_table.keys():
        logging.info('Looking-up the term \'{:s}\''.format(src_term))
        sense_to_cluster_table[src_term] = dict()
        for cluster_id in sense_clusters_table[src_term].keys():
            # Construct cluster-ID lookup table entry
            for tgt_sense in sense_clusters_table[src_term][cluster_id]['[SENSES]']:
                # Lemmatizing single words is not ideal, but is expected to improve attractor recall
                _, _, tgt_lemmas, _, _, _, _, _ = \
                    _process_strings(tgt_sense, tgt_nlp, True, False, False, False, False)
                # Multi-word targets are excluded for simplicity (as a result, some words are dropped)
                if len(tgt_lemmas) < 1:
                    continue
                tgt_lemma = tgt_lemmas[0]
                if len(tgt_lemma) > 0:
                    if not sense_to_cluster_table[src_term].get(tgt_lemma, None):
                        sense_to_cluster_table[src_term][tgt_lemma] = [(tgt_sense, cluster_id, True)]
                    else:
                        sense_to_cluster_table[src_term][tgt_lemma].append((tgt_sense, cluster_id, True))

            # Check for blacklisted and ambiguous senses
            senses_to_ignore = list()
            senses_to_ignore += sense_clusters_table[src_term][cluster_id].get('[BLACKLISTED SENSES]', [])
            senses_to_ignore += sense_clusters_table[src_term][cluster_id].get('[AMBIGUOUS SENSES]', [])
            for tgt_sense in senses_to_ignore:
                # Lemmatizing single words is not ideal, but is expected to improve attractor recall
                _, _, tgt_lemmas, _, _, _, _, _ = \
                    _process_strings(tgt_sense, tgt_nlp, True, False, False, False, False)
                # Multi-word targets are excluded for simplicity (as a result, some words are dropped)
                if len(tgt_lemmas) < 1:
                    continue
                tgt_lemma = tgt_lemmas[0]
                if len(tgt_lemma) > 0:
                    if not sense_to_cluster_table[src_term].get(tgt_lemma, None):
                        sense_to_cluster_table[src_term][tgt_lemma] = [(tgt_sense, cluster_id, False)]
                    else:
                        sense_to_cluster_table[src_term][tgt_lemma].append((tgt_sense, cluster_id, False))
    return sense_to_cluster_table


def evaluate_attack_success(json_challenge_set_path,
                            source_sentences_path,
                            translations_path,
                            adversarial_source_sentences_path,
                            adversarial_translations_path,
                            alignments_path,
                            adversarial_alignments_path,
                            attractors_path,
                            sense_clusters_path,
                            output_dir):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    """ Detects successful attacks (natural and adversarial) and computes correlation between attack success and
    various metrics. """

    def _score_and_filter(challenge_entry, attractors_entry, ambiguous_term, ambiguous_form, idx):
        """ Helper function for evaluating challenge samples. """

        sense_lemmas_to_cluster = sense_lemmas_to_cluster_table.get(ambiguous_term, None)
        sense_tokens_to_cluster = sense_tokens_to_cluster_table.get(ambiguous_term, None)
        cluster_to_sense_lemmas = cluster_to_sense_lemmas_table.get(ambiguous_term, None)
        cluster_to_sense_tokens = cluster_to_sense_tokens_table.get(ambiguous_term, None)

        for true_cluster in challenge_entry.keys():
            for adv_cluster in challenge_entry[true_cluster].keys():
                for sample_entry in challenge_entry[true_cluster][adv_cluster]:
                    idx += 1
                    # Unpack
                    adv_src = sample_entry[0]
                    true_src = sample_entry[1]
                    true_tgt = sample_entry[2]
                    attractor = sample_entry[3]
                    ambiguous_token_ws_loc_adv = sample_entry[5]
                    ambiguous_token_loc_seed = sample_entry[7]
                    ambiguous_token_ws_loc_seed = sample_entry[8]  # list
                    attractor_tokens_ws_loc = sample_entry[11]  # list
                    seed_cluster_id = sample_entry[12]
                    adv_cluster_id = sample_entry[13]
                    seed_cluster_senses = sample_entry[14]
                    adv_cluster_senses = sample_entry[15]

                    # Ignore samples containing multiple instances of the attractor term
                    true_src_tokens = [tok.strip(punctuation_plus_space) for tok in true_src.split()]
                    true_src_tokens = [tok for tok in true_src_tokens if len(tok) > 0]
                    adv_src_tokens = [tok.strip(punctuation_plus_space) for tok in adv_src.split()]
                    adv_src_tokens = [tok for tok in adv_src_tokens if len(tok) > 0]
                    if adv_src_tokens.count(attractor.strip(punctuation_plus_space)) > 1:
                        continue
                    # Ignore samples with sentence-initial attractors
                    if 0 in attractor_tokens_ws_loc:
                        continue
                    # Ignore short sentences
                    if len(true_src_tokens) < 10:
                        continue

                    # Look up translations and alignments
                    # true_challenge_sentence_id = true_sources[true_src.strip()]
                    true_challenge_sentence_id = idx
                    true_translation = true_translations[true_challenge_sentence_id]
                    true_alignments = true_alignment_table[true_challenge_sentence_id] if \
                        true_alignment_table is not None else None

                    # used to debug whether there were actual differences :)
                    # if true_challenge_sentence_id != idx:
                    #     logging.warning(f"{true_challenge_sentence_id}, {idx}, {true_src}")
                    #     flag1 = true_translations[idx] == true_translations[true_challenge_sentence_id]
                    #     flag2 = true_alignment_table[idx] == true_alignment_table[true_challenge_sentence_id]
                    #     logging.warning(f"{flag1} - {flag2}")

                    adv_translation, adv_alignments = None, None
                    if adv_sources is not None:
                        # adv_challenge_sentence_id = adv_sources[adv_src.strip()]
                        adv_challenge_sentence_id = idx
                        adv_translation = adv_translations[adv_challenge_sentence_id]
                        adv_alignments = adv_alignment_table[adv_challenge_sentence_id] if \
                            adv_alignment_table is not None else None

                    # Look up sense mappings
                    ambiguous_token = ambiguous_form if ambiguous_form is not None else ambiguous_term

                    other_cluster_ids = list(attractors_entry.keys())
                    other_cluster_ids.pop(other_cluster_ids.index(seed_cluster_id))
                    try:
                        other_cluster_ids.pop(other_cluster_ids.index(adv_cluster_id))
                    except ValueError:
                        pass

                    # Get NMT labels
                    true_nmt_label, true_translation_sense_clusters, _ = \
                        _get_nmt_label(true_translation,
                                       adv_src,
                                       true_src,
                                       ambiguous_token,
                                       ambiguous_token_ws_loc_seed,
                                       ambiguous_token_ws_loc_adv,
                                       attractor_tokens_ws_loc,
                                       seed_cluster_id,
                                       adv_cluster_id,
                                       other_cluster_ids,
                                       sense_lemmas_to_cluster,
                                       cluster_to_sense_lemmas,
                                       sense_tokens_to_cluster,
                                       cluster_to_sense_tokens,
                                       true_alignments,
                                       is_adv=False)

                    if adv_translations is not None:
                        adv_nmt_label, adv_translation_sense_clusters, attractor_is_translated = \
                            _get_nmt_label(adv_translation,
                                           adv_src,
                                           true_src,
                                           ambiguous_token,
                                           ambiguous_token_ws_loc_seed,
                                           ambiguous_token_ws_loc_adv,
                                           attractor_tokens_ws_loc,
                                           seed_cluster_id,
                                           adv_cluster_id,
                                           other_cluster_ids,
                                           sense_lemmas_to_cluster,
                                           cluster_to_sense_lemmas,
                                           sense_tokens_to_cluster,
                                           cluster_to_sense_tokens,
                                           adv_alignments,
                                           is_adv=True,
                                           true_translation=true_translation,
                                           true_alignments=true_alignments)

                        # Assemble table entry
                        new_table_entry = [true_src,
                                           true_translation,
                                           adv_src,
                                           adv_translation,
                                           true_tgt,
                                           ambiguous_token,
                                           attractor,
                                           seed_cluster_id,
                                           adv_cluster_id,
                                           seed_cluster_senses,
                                           adv_cluster_senses]

                    else:
                        adv_nmt_label, adv_translation_sense_clusters, attractor_is_translated, new_table_entry = \
                            None, None, None, None

                    # Sort true samples into appropriate output tables, based on filtering outcome
                    seed_table_to_expand = None
                    if true_nmt_label == 'not_flipped':
                        stat_dict['num_adversarial_samples'][0] += 1
                        stat_dict.setdefault('not_flipped_indices', []).append(idx)
                        if not unique_samples.get((true_src, ambiguous_token, ambiguous_token_loc_seed), None):
                            stat_dict['num_true_samples_good_translations'] += 1
                            unique_samples[(true_src, ambiguous_token, ambiguous_token_loc_seed)] = True
                            seed_table_to_expand = true_samples_good_translations
                        label_id = 0
                    elif true_nmt_label == 'flipped_to_attr':
                        stat_dict['num_adversarial_samples'][1] += 1
                        stat_dict.setdefault('flipped_to_attr_indices', []).append(idx)
                        if not unique_samples.get((true_src, ambiguous_token, ambiguous_token_loc_seed), None):
                            stat_dict['num_true_samples_bad_translations'] += 1
                            unique_samples[(true_src, ambiguous_token, ambiguous_token_loc_seed)] = True
                            seed_table_to_expand = true_samples_bad_translations
                        label_id = 1
                    else:
                        stat_dict['num_adversarial_samples'][2] += 1
                        stat_dict.setdefault('flipped_to_other_indices', []).append(idx)
                        if not unique_samples.get((true_src, ambiguous_token, ambiguous_token_loc_seed), None):
                            stat_dict['num_other_wsd_errors_seed'] += 1
                            unique_samples[(true_src, ambiguous_token, ambiguous_token_loc_seed)] = True
                            seed_table_to_expand = other_wsd_errors_seed
                        label_id = 2

                    adv_table_to_expand = None
                    if adv_translations is not None:
                        # Sort adversarial samples
                        if adv_nmt_label == 'not_flipped':
                            stat_dict['num_not_flipped'][label_id] += 1
                            adv_table_to_expand = not_flipped_adv_samples
                        elif adv_nmt_label == 'flipped_to_attr':
                            stat_dict['num_flipped_to_attr_sense'][label_id] += 1
                            adv_table_to_expand = flipped_to_attr_sense_adv_samples
                        elif adv_nmt_label == 'flipped_to_other':
                            stat_dict['num_flipped_to_other_sense'][label_id] += 1
                            adv_table_to_expand = flipped_to_other_sense_adv_samples
                        else:
                            stat_dict['num_other_wsd_errors_adv'][label_id] += 1
                            adv_table_to_expand = other_wsd_errors_adv

                    # Collect seed translations
                    if seed_table_to_expand is not None:
                        if not seed_table_to_expand.get(ambiguous_term, None):
                            seed_table_to_expand[ambiguous_term] = dict()
                        if not seed_table_to_expand[ambiguous_term].get(true_cluster, None):
                            seed_table_to_expand[ambiguous_term][true_cluster] = dict()
                        if not seed_table_to_expand[ambiguous_term][true_cluster].get(adv_cluster, None):
                            seed_table_to_expand[ambiguous_term][true_cluster][adv_cluster] = dict()
                        if not seed_table_to_expand[ambiguous_term][true_cluster][adv_cluster].get(true_src, None):
                            seed_table_to_expand[ambiguous_term][true_cluster][adv_cluster][true_src] = list()
                        seed_table_to_expand[ambiguous_term][true_cluster][adv_cluster][true_src]\
                            .append([true_translation, true_tgt, true_translation_sense_clusters,
                                     ambiguous_token_loc_seed])

                    # Collect attack success samples
                    if adv_table_to_expand is not None:
                        if not adv_table_to_expand.get(ambiguous_term, None):
                            adv_table_to_expand[ambiguous_term] = dict()
                        if not adv_table_to_expand[ambiguous_term].get(true_cluster, None):
                            adv_table_to_expand[ambiguous_term][true_cluster] = dict()
                        if not adv_table_to_expand[ambiguous_term][true_cluster].get(adv_cluster, None):
                            adv_table_to_expand[ambiguous_term][true_cluster][adv_cluster] = dict()
                        if not adv_table_to_expand[ambiguous_term][true_cluster][adv_cluster].get(true_src, None):
                            adv_table_to_expand[ambiguous_term][true_cluster][adv_cluster][true_src] = list()
                        adv_table_to_expand[ambiguous_term][true_cluster][adv_cluster][true_src].append(new_table_entry)

        return idx

    def _show_stats():
        """ Helper for reporting on the generation process. """

        def _pc(enum, denom):
            """ Helper function for computing percentage of the evaluated sample type """
            return (enum / denom) * 100 if denom > 0 else 0

        logging.info('-' * 20)

        num_all_seed_samples = \
            stat_dict['num_true_samples_good_translations'] + \
            stat_dict['num_true_samples_bad_translations'] + \
            stat_dict['num_other_wsd_errors_seed']
        num_all_incorrect_adv_translations = \
            sum(stat_dict['num_flipped_to_attr_sense']) + \
            sum(stat_dict['num_flipped_to_other_sense']) + \
            sum(stat_dict['num_other_wsd_errors_adv'])
        # ==============================================================================================================
        if adv_translations is None:
            logging.info('Evaluated {:d} seed samples'.format(num_all_seed_samples))
            logging.info('{:d} ({:.4f}%) seed sentences have been translated CORRECTLY'
                         .format(stat_dict['num_true_samples_good_translations'],
                                 _pc(stat_dict['num_true_samples_good_translations'], num_all_seed_samples)))

            logging.info('{:d} ({:.4f}%) seed sentences have been translated INCORRECTLY with the homograph flipped '
                         'to the most likely incorrect sense cluster '
                         .format(stat_dict['num_true_samples_bad_translations'],
                                 _pc(stat_dict['num_true_samples_bad_translations'], num_all_seed_samples)))

            logging.info('{:d} ({:.4f}%) seed sentences have been translated INCORRECTLY in total'
                         .format(stat_dict['num_true_samples_bad_translations'] +
                                 stat_dict['num_other_wsd_errors_seed'],
                                 _pc(stat_dict['num_true_samples_bad_translations'] +
                                     stat_dict['num_other_wsd_errors_seed'], num_all_seed_samples)))
            # ==============================================================================================================
        else:
            logging.info('-' * 20)
            num_all_adversarial_samples = sum(stat_dict['num_adversarial_samples'])
            num_all_not_flipped = sum(stat_dict['num_not_flipped'])
            logging.info('{:d} ({:.4f}%) adversarial sentences have been translated CORRECTLY'
                         .format(num_all_not_flipped,
                                 _pc(num_all_not_flipped, num_all_adversarial_samples)))

            logging.info('{:d} ({:.4f}%) adversarial sentences have been translated INCORRECTLY'
                         .format(num_all_incorrect_adv_translations,
                                 _pc(num_all_incorrect_adv_translations, num_all_adversarial_samples)))
            # ==========================================================================================================
            logging.info('-' * 20)
            num_all_flipped_to_attr = sum(stat_dict['num_flipped_to_attr_sense'])
            logging.info('{:d} ({:.4f}%) / ({:.4f}%) INCORRECT adversarial translations were flipped to the '
                         'ATTRACTOR\'S target sense'
                         .format(num_all_flipped_to_attr,
                                 _pc(num_all_flipped_to_attr, num_all_incorrect_adv_translations),
                                 _pc(num_all_flipped_to_attr, num_all_adversarial_samples)))
            logging.info('===> Of those, {:d} ({:.4f}%) have correct seed translations (i.e. successful attacks, as reported in Fugure 3 of the paper) <=='
                         .format(stat_dict['num_flipped_to_attr_sense'][0],
                                 _pc(stat_dict['num_flipped_to_attr_sense'][0],
                                     stat_dict['num_adversarial_samples'][0])))
            # ==========================================================================================================
            logging.info('-' * 20)
            num_all_flipped_to_other = sum(stat_dict['num_flipped_to_other_sense'])
            logging.info('{:d} ({:.4f}%) / ({:.4f}%) INCORRECT adversarial translations were flipped to some '
                         'OTHER KNOWN target sense'
                         .format(num_all_flipped_to_other,
                                 _pc(num_all_flipped_to_other, num_all_incorrect_adv_translations),
                                 _pc(num_all_flipped_to_other, num_all_adversarial_samples)))
            logging.info('Of those, {:d} ({:.4f}%) have correct seed translations'
                         .format(stat_dict['num_flipped_to_other_sense'][0],
                                 _pc(stat_dict['num_flipped_to_other_sense'][0],
                                     stat_dict['num_adversarial_samples'][0])))

    # Read-in adversarial samples
    logging.info('Reading-in JSON samples table ...')
    with open(json_challenge_set_path, 'r', encoding='utf8') as asp:
        challenge_samples_table = json.load(asp)

    # Read-in attractor table
    logging.info('Reading-in attractor table ...')
    with open(attractors_path, 'r', encoding='utf8') as ap:
        attractors_table = json.load(ap)

    # Load sense to cluster mappings
    logging.info('Reading-in sense clusters ...')
    with open(sense_clusters_path, 'r', encoding='utf8') as scp:
        sense_clusters_table = json.load(scp)

    # Add blacklisted, ambiguous terms to the set of acceptable translations
    for term in sense_clusters_table.keys():
        for cluster in sense_clusters_table[term].keys():
            if '[AMBIGUOUS SENSES]' in sense_clusters_table[term][cluster].keys():
                sense_clusters_table[term][cluster]['[SENSES]'] += \
                    sense_clusters_table[term][cluster]['[AMBIGUOUS SENSES]']
                sense_clusters_table[term][cluster].pop('[AMBIGUOUS SENSES]')
            if '[BLACKLISTED SENSES]' in sense_clusters_table[term][cluster].keys():
                sense_clusters_table[term][cluster]['[SENSES]'] += \
                    sense_clusters_table[term][cluster]['[BLACKLISTED SENSES]']
                sense_clusters_table[term][cluster].pop('[BLACKLISTED SENSES]')
            sense_clusters_table[term][cluster]['[SENSES]'] = \
                list(set(sense_clusters_table[term][cluster]['[SENSES]']))
    sense_to_cluster_table = _build_cluster_lookup(sense_clusters_table)

    # Post-process sense-to-cluster table
    sense_lemmas_to_cluster_table = sense_to_cluster_table
    sense_tokens_to_cluster_table = dict()
    for src_term in sense_lemmas_to_cluster_table.keys():
        sense_tokens_to_cluster_table[src_term] = dict()
        for cluster_tuple_list in sense_lemmas_to_cluster_table[src_term].values():
            for cluster_tuple in cluster_tuple_list:
                sense_tokens_to_cluster_table[src_term][cluster_tuple[0].lower()] = [cluster_tuple]

    # Derive a cluster-to-sense-tokens table, used for compound analysis
    cluster_to_sense_lemmas_table = dict()
    for src_term in sense_lemmas_to_cluster_table.keys():
        cluster_to_sense_lemmas_table[src_term] = dict()
        for sense, clusters in sense_lemmas_to_cluster_table[src_term].items():
            for cls_tpl in clusters:
                if not cluster_to_sense_lemmas_table[src_term].get(cls_tpl[1], None):
                    cluster_to_sense_lemmas_table[src_term][cls_tpl[1]] = [sense]
                else:
                    if sense not in cluster_to_sense_lemmas_table[src_term][cls_tpl[1]]:
                        cluster_to_sense_lemmas_table[src_term][cls_tpl[1]].append(sense)

    # Derive a cluster-to-sense-lemmas table, used for compound analysis
    cluster_to_sense_tokens_table = dict()
    for src_term in sense_tokens_to_cluster_table.keys():
        cluster_to_sense_tokens_table[src_term] = dict()
        for sense, clusters in sense_lemmas_to_cluster_table[src_term].items():
            for cls_tpl in clusters:
                if not cluster_to_sense_tokens_table[src_term].get(cls_tpl[1], None):
                    cluster_to_sense_tokens_table[src_term][cls_tpl[1]] = [cls_tpl[0]]
                else:
                    if sense not in cluster_to_sense_tokens_table[src_term][cls_tpl[1]]:
                        cluster_to_sense_tokens_table[src_term][cls_tpl[1]].append(cls_tpl[0])

    # Restructure NMT translations for easier access
    logging.info('-' * 10)
    logging.info('Hashing true translations ...')
    with open(source_sentences_path, 'r', encoding='utf8') as tsp:
        true_sources = {line.strip(): line_id for line_id, line in enumerate(tsp)}
    with open(translations_path, 'r', encoding='utf8') as ttp:
        true_translations = {line_id: line.strip() for line_id, line in enumerate(ttp)}

    if adversarial_translations_path is not None:
        logging.info('Hashing adversarial translations ...')
        with open(adversarial_source_sentences_path, 'r', encoding='utf8') as asp:
            adv_sources = {line.strip(): line_id for line_id, line in enumerate(asp)}
        with open(adversarial_translations_path, 'r', encoding='utf8') as atp:
            adv_translations = {line_id: line for line_id, line in enumerate(atp)}
        logging.info('-' * 10)
    else:
        adv_sources = None
        adv_translations = None

    # Load alignment file, if available
    true_alignment_table = None
    adv_alignment_table = None
    if alignments_path:
        with open(alignments_path, 'r', encoding='utf8') as tap:
            true_alignment_table = {line_id: line for line_id, line in enumerate(tap)}
    if adversarial_alignments_path:
        with open(adversarial_alignments_path, 'r', encoding='utf8') as aap:
            adv_alignment_table = {line_id: line for line_id, line in enumerate(aap)}

    # For stats
    unique_samples = dict()

    # Seed translations
    true_samples_good_translations = dict()
    true_samples_bad_translations = dict()
    other_wsd_errors_seed = dict()

    # Attack success tables
    flipped_to_attr_sense_adv_samples = dict()
    flipped_to_other_sense_adv_samples = dict()
    other_wsd_errors_adv = dict()
    not_flipped_adv_samples = dict()

    # Initialize variables for reporting
    stat_dict = {
        'num_true_samples_good_translations': 0,
        'num_true_samples_bad_translations': 0,
        'num_other_wsd_errors_seed': 0,

        'num_adversarial_samples': [0, 0, 0],
        'num_flipped_to_attr_sense': [0, 0, 0],
        'num_flipped_to_other_sense': [0, 0, 0],
        'num_other_wsd_errors_adv': [0, 0, 0],
        'num_not_flipped': [0, 0, 0]
    }

    idx = -1

    # Iterate over challenge samples
    for term_id, term in enumerate(challenge_samples_table.keys()):

        logging.info('Looking-up the term \'{:s}\''.format(term))
        # Apply adversarial filtering and compute LM-based fluency / acceptability scores
        idx = _score_and_filter(challenge_samples_table[term], attractors_table[term], term, None, idx)

        # Occasionally report statistics
        if term_id > 0 and term_id % 10 == 0:
            logging.info('Looked up {:d} terms; reporting intermediate statistics:'.format(term_id))
            _show_stats()

    # Final report
    logging.info('\n\n')
    logging.info('Looked up {:d} terms; reporting FINAL statistics:'.format(len(challenge_samples_table.keys())))
    _show_stats()

    # Construct output paths
    table_list = [
        stat_dict,

        true_samples_good_translations,
        true_samples_bad_translations,
        other_wsd_errors_seed,

        flipped_to_attr_sense_adv_samples,
        flipped_to_other_sense_adv_samples,
        other_wsd_errors_adv,
        not_flipped_adv_samples
    ]

    path_list = [
        os.path.join(output_dir, 'stats'),

        os.path.join(output_dir, 'true_samples_good_translations.{:s}'.format(src_lang)),
        os.path.join(output_dir, 'true_samples_bad_translations.{:s}'.format(src_lang)),
        os.path.join(output_dir, 'other_wsd_errors_seed.{:s}'.format(src_lang)),

        os.path.join(output_dir, 'flipped_to_attr_sense_adv_samples.{:s}'.format(src_lang)),
        os.path.join(output_dir, 'flipped_to_other_sense_adv_samples.{:s}'.format(src_lang)),
        os.path.join(output_dir, 'other_wsd_errors_adv.{:s}'.format(src_lang)),
        os.path.join(output_dir, 'not_flipped_adv_samples.{:s}'.format(src_lang))
    ]

    # Save
    for table, path in zip(table_list, path_list):
        with open(path + '.json', 'w', encoding='utf8') as json_file:
            json.dump(table, json_file, indent=3, sort_keys=True, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_challenge_set_path', type=str, required=True,
                        help='path to the JSON file containing challenge samples (natural or adversarial)')
    parser.add_argument('--source_sentences_path', type=str, required=True,
                        help='path to the source sentences given to the evaluated NMT model')
    parser.add_argument('--translations_path', type=str, required=True,
                        help='path to the translations produced by the evaluated NMT model for the source sentences')
    parser.add_argument('--alignments_path', type=str, default=None,
                        help='path to file containing the fastalign alignments for the translations of the original '
                             'challenge sentences (OPTIONAL)')
    parser.add_argument('--adversarial_source_sentences_path', type=str, default=None,
                        help='path to the perturbed source sentences given to the evaluated NMT model (OPTIONAL)')
    parser.add_argument('--adversarial_translations_path', type=str, default=None,
                        help='path to the translations produced by the evaluated NMT model for the adversarially '
                             'perturbed source sentences (OPTIONAL)')
    parser.add_argument('--adversarial_alignments_path', type=str, default=None,
                        help='path to file containing the fastalign alignments for the translations of the '
                             'adversarially perturbed challenge sentences (OPTIONAL)')
    parser.add_argument('--attractors_path', type=str, required=True,
                        help='path to the JSON file containing the extracted attractor terms')
    parser.add_argument('--sense_clusters_path', type=str, default=None,
                        help='path to the JSON file containing scraped BabelNet sense clusters')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='path to the directory to which the incorrectly translated '
                             'challenge samples will be saved for manual evaluation')
    parser.add_argument('--lang_pair', type=str, default=None,
                        help='language pair of the bitext; expected format is src-tgt')
    args = parser.parse_args()

    # Logging to file
    base_dir = '/'.join(args.source_sentences_path.split('/')[:-1])
    file_name = args.source_sentences_path.split('/')[-1]
    file_name = '.'.join(file_name.split('.')[:-1])
    file_name = file_name if len(file_name) > 0 else 'log'
    log_dir = '{:s}/logs/'.format(base_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_path = '{:s}{:s}.log'.format(log_dir, file_name)
    logging.basicConfig(filename=log_path, filemode='w', level=logging.INFO, format='%(levelname)s: %(message)s')
    # Logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    # Instantiate processing pipeline
    src_lang, tgt_lang = args.lang_pair.strip().split('-')
    spacy_map = {'en': 'en_core_web_sm', 'de': 'de_core_news_sm'}
    try:
        src_nlp = spacy.load(spacy_map[src_lang], disable=['parser', 'ner', 'textcat'])
        tgt_nlp = spacy.load(spacy_map[tgt_lang], disable=['parser', 'ner', 'textcat'])
    except KeyError:
        logging.info('SpaCy does not support the language {:s} or {:s}. Exiting.'.format(src_lang, tgt_lang))
        sys.exit(0)
    pct_stripper = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    punctuation_plus_space = string.punctuation + ' ' + '\t' + '\n'
    # Import stopword list
    if src_lang == 'en':
        STOP_WORDS = stopwords.words('english')
    else:
        STOP_WORDS = []

    evaluate_attack_success(args.json_challenge_set_path,
                            args.source_sentences_path,
                            args.translations_path,
                            args.adversarial_source_sentences_path,
                            args.adversarial_translations_path,
                            args.alignments_path,
                            args.adversarial_alignments_path,
                            args.attractors_path,
                            args.sense_clusters_path,
                            args.output_dir)