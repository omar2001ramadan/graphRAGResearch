import re
import json
import sys
import csv
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_data(html_content):
    # Adjusted regex patterns to extract nodes and edges
    nodes_pattern = re.compile(r'nodes\s*=\s*new\s+vis\.DataSet\(\s*(\[.*?\])\s*\);', re.DOTALL)
    edges_pattern = re.compile(r'edges\s*=\s*new\s+vis\.DataSet\(\s*(\[.*?\])\s*\);', re.DOTALL)
    
    # Find nodes and edges data
    nodes_match = nodes_pattern.search(html_content)
    edges_match = edges_pattern.search(html_content)
    
    if not nodes_match or not edges_match:
        print("Could not find nodes or edges in the provided HTML content.")
        sys.exit(1)
    
    # Load nodes and edges as JSON
    nodes_json = json.loads(nodes_match.group(1))
    edges_json = json.loads(edges_match.group(1))
    
    return nodes_json, edges_json

def parse_entities(nodes_json):
    entities = []
    for node in nodes_json:
        label = node.get('label', '')
        label_match = re.match(r'^(.*?)\s+<(.+?)>$', label)
        if label_match:
            text = label_match.group(1).strip()
            entity_type = label_match.group(2).strip()
            # Normalize text: lowercase and strip
            text_normalized = text.lower().strip()
            entity_type_normalized = entity_type.lower().strip()
            entities.append({
                'text': text_normalized,
                'type': entity_type_normalized,
                'id': node['id']
            })
    return entities

def parse_relations(edges_json):
    relations = []
    for edge in edges_json:
        from_node = edge.get('from')
        to_node = edge.get('to')
        label = edge.get('label', '').strip()
        # Normalize label
        label_normalized = label.lower().strip()
        relations.append({
            'from': from_node,
            'to': to_node,
            'label': label_normalized
        })
    return relations

def compute_similarity(*texts):
    # Normalize texts: lowercase and strip
    texts = [text.lower().strip() for text in texts]
    vectorizer = TfidfVectorizer().fit(texts)
    tfidf = vectorizer.transform(texts)
    # Compute pairwise cosine similarities and take the average
    similarities = []
    for i in range(len(texts) - 1):
        sim = cosine_similarity(tfidf[i:i+1], tfidf[i+1:i+2])[0][0]
        similarities.append(sim)
    return sum(similarities) / len(similarities) if similarities else 0

def compare_entities(entities_a, entities_b, threshold=0.7):
    matched = []
    extra = entities_b.copy()
    missing = entities_a.copy()
    used_indices_b = set()

    for idx_a, entity_a in enumerate(entities_a):
        best_match = None
        best_score = 0
        best_idx_b = -1
        for idx_b, entity_b in enumerate(entities_b):
            if idx_b in used_indices_b:
                continue  # Skip entities already matched
            # Concatenate text and type for comparison
            entity_a_str = f"{entity_a['text']} {entity_a['type']}"
            entity_b_str = f"{entity_b['text']} {entity_b['type']}"
            sim_score = compute_similarity(entity_a_str, entity_b_str)
            if sim_score > best_score:
                best_score = sim_score
                best_match = entity_b
                best_idx_b = idx_b
        if best_score >= threshold and best_match is not None:
            matched.append((entity_a, best_match, best_score))
            used_indices_b.add(best_idx_b)
            extra.remove(best_match)
            missing.remove(entity_a)
    return matched, extra, missing

def compare_relations(relations_a, relations_b, entities_a, entities_b, threshold=0.7):
    matched = []
    extra = relations_b.copy()
    missing = relations_a.copy()
    used_relations_b = set()

    # Build mapping from IDs to entity texts for entities in both graphs
    entity_text_a = {entity['id']: entity['text'] for entity in entities_a}
    entity_text_b = {entity['id']: entity['text'] for entity in entities_b}

    for idx_a, rel_a in enumerate(relations_a):
        best_match_idx = None
        best_score = 0
        for idx_b, rel_b in enumerate(relations_b):
            if idx_b in used_relations_b:
                continue  # Skip relations already matched
            # Compare from and to nodes using their text labels
            from_a_text = entity_text_a.get(rel_a['from'], '').lower()
            to_a_text = entity_text_a.get(rel_a['to'], '').lower()
            from_b_text = entity_text_b.get(rel_b['from'], '').lower()
            to_b_text = entity_text_b.get(rel_b['to'], '').lower()
            sim_from = compute_similarity(from_a_text, from_b_text)
            sim_to = compute_similarity(to_a_text, to_b_text)
            sim_label = compute_similarity(rel_a['label'], rel_b['label'])
            avg_score = (sim_from + sim_to + sim_label) / 3
            if avg_score > best_score:
                best_score = avg_score
                best_match_idx = idx_b
        if best_score >= threshold and best_match_idx is not None:
            matched.append((rel_a, relations_b[best_match_idx], best_score))
            used_relations_b.add(best_match_idx)
            extra.remove(relations_b[best_match_idx])
            missing.remove(rel_a)
    return matched, extra, missing

def main():
    if len(sys.argv) != 5:
        print("Usage: python Compare_Graphs.py <path_to_true_graph> <path_to_generated_graph1> <path_to_generated_graph2> <path_to_generated_graph3>")
        sys.exit(1)
    
    true_graph_path = sys.argv[1]
    generated_graph_paths = sys.argv[2:5]
    
    # Read HTML contents from the truth graph
    with open(true_graph_path, 'r', encoding='utf-8') as f:
        html_content_a = f.read()
    
    # Extract data from truth graph
    nodes_a, edges_a = extract_data(html_content_a)
    
    # Parse entities and relations from truth graph
    entities_a = parse_entities(nodes_a)
    relations_a = parse_relations(edges_a)
    
    total_entities_truth = len(entities_a)
    total_relations_truth = len(relations_a)
    
    # Lists to store metrics for each trial
    entity_metrics = []
    relation_metrics = []
    detailed_analysis = []
    
    # Process each generated graph
    for idx, generated_graph_path in enumerate(generated_graph_paths):
        # Read HTML contents from generated graph
        with open(generated_graph_path, 'r', encoding='utf-8') as f:
            html_content_b = f.read()
        
        # Extract data from generated graph
        nodes_b, edges_b = extract_data(html_content_b)
        
        # Parse entities and relations from generated graph
        entities_b = parse_entities(nodes_b)
        relations_b = parse_relations(edges_b)
        
        # Compare entities
        matched_entities, extra_entities, missing_entities = compare_entities(entities_a, entities_b)
        
        # Compare relations
        matched_relations, extra_relations, missing_relations = compare_relations(relations_a, relations_b, entities_a, entities_b)
        
        # Compute metrics for this trial
        total_entities_generated = len(entities_b)
        total_relations_generated = len(relations_b)
        matched_entities_count = len(matched_entities)
        extra_entities_count = len(extra_entities)
        missing_entities_count = len(missing_entities)
        entity_coverage = (matched_entities_count / total_entities_truth) * 100 if total_entities_truth > 0 else 0
        
        matched_relations_count = len(matched_relations)
        extra_relations_count = len(extra_relations)
        missing_relations_count = len(missing_relations)
        relation_coverage = (matched_relations_count / total_relations_truth) * 100 if total_relations_truth > 0 else 0
        
        # Store metrics
        entity_metrics.append({
            'Total Entities in Truth Graph': total_entities_truth,
            'Total Entities in Generated Graph': total_entities_generated,
            'Matched Entities': matched_entities_count,
            'Extra Entities': extra_entities_count,
            'Missing Entities': missing_entities_count,
            'Entity Coverage (%)': entity_coverage
        })
        
        relation_metrics.append({
            'Total Relations in Truth Graph': total_relations_truth,
            'Total Relations in Generated Graph': total_relations_generated,
            'Matched Relations': matched_relations_count,
            'Extra Relations': extra_relations_count,
            'Missing Relations': missing_relations_count,
            'Relationship Coverage (%)': relation_coverage
        })
        
        # Store detailed analysis for this trial
        detailed_analysis.append({
            'trial': idx + 1,
            'matched_entities': matched_entities,
            'extra_entities': extra_entities,
            'missing_entities': missing_entities,
            'matched_relations': matched_relations,
            'extra_relations': extra_relations,
            'missing_relations': missing_relations,
            'entities_a': entities_a,
            'entities_b': entities_b
        })
    
    # Compute average metrics
    avg_entity_metrics = {key: sum(d[key] for d in entity_metrics) / len(entity_metrics) for key in entity_metrics[0]}
    avg_relation_metrics = {key: sum(d[key] for d in relation_metrics) / len(relation_metrics) for key in relation_metrics[0]}
    
    # Output results to CSV
    with open('comparison_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        # Write header
        csvwriter.writerow(['Comparison Results'])
        csvwriter.writerow([])
        
        # Write metrics for each trial
        for idx in range(len(generated_graph_paths)):
            csvwriter.writerow([f'Comparison with Generated Graph {idx + 1}'])
            csvwriter.writerow([])
            
            csvwriter.writerow(['Entity Comparison Summary'])
            for key, value in entity_metrics[idx].items():
                csvwriter.writerow([key, f"{value:.2f}" if isinstance(value, float) else value])
            csvwriter.writerow([])
            
            csvwriter.writerow(['Relation Comparison Summary'])
            for key, value in relation_metrics[idx].items():
                csvwriter.writerow([key, f"{value:.2f}" if isinstance(value, float) else value])
            csvwriter.writerow([])
            csvwriter.writerow([])
        
        # Write average metrics
        csvwriter.writerow(['Average Metrics Across All Trials'])
        csvwriter.writerow([])
        csvwriter.writerow(['Entity Comparison Summary'])
        for key, value in avg_entity_metrics.items():
            csvwriter.writerow([key, f"{value:.2f}" if isinstance(value, float) else value])
        csvwriter.writerow([])
        csvwriter.writerow(['Relation Comparison Summary'])
        for key, value in avg_relation_metrics.items():
            csvwriter.writerow([key, f"{value:.2f}" if isinstance(value, float) else value])
        csvwriter.writerow([])
        
        # In-depth Analysis
        csvwriter.writerow(['In-Depth Analysis of Included and Left-Out Items'])
        csvwriter.writerow([])
        
        for analysis in detailed_analysis:
            trial = analysis['trial']
            entities_a = analysis['entities_a']
            entities_b = analysis['entities_b']
            entities_a_dict = {entity['id']: entity for entity in entities_a}
            entities_b_dict = {entity['id']: entity for entity in entities_b}
            
            csvwriter.writerow([f'Detailed Analysis for Generated Graph {trial}'])
            csvwriter.writerow([])
            
            # Matched Entities
            csvwriter.writerow(['Matched Entities'])
            csvwriter.writerow(['Truth Entity Text', 'Truth Entity Type', 'Generated Entity Text', 'Generated Entity Type', 'Similarity Score'])
            for entity_a, entity_b, sim_score in analysis['matched_entities']:
                csvwriter.writerow([
                    entity_a['text'], entity_a['type'],
                    entity_b['text'], entity_b['type'],
                    f"{sim_score:.2f}"
                ])
            csvwriter.writerow([])
            
            # Extra Entities
            csvwriter.writerow(['Extra Entities in Generated Graph'])
            csvwriter.writerow(['Entity Text', 'Entity Type'])
            for entity in analysis['extra_entities']:
                csvwriter.writerow([entity['text'], entity['type']])
            csvwriter.writerow([])
            
            # Missing Entities
            csvwriter.writerow(['Missing Entities from Generated Graph'])
            csvwriter.writerow(['Entity Text', 'Entity Type'])
            for entity in analysis['missing_entities']:
                csvwriter.writerow([entity['text'], entity['type']])
            csvwriter.writerow([])
            
            # Matched Relations
            csvwriter.writerow(['Matched Relations'])
            csvwriter.writerow(['Truth From', 'Truth Label', 'Truth To', 'Generated From', 'Generated Label', 'Generated To', 'Similarity Score'])
            for rel_a, rel_b, avg_score in analysis['matched_relations']:
                from_a_text = entities_a_dict.get(rel_a['from'], {'text': ''})['text']
                to_a_text = entities_a_dict.get(rel_a['to'], {'text': ''})['text']
                from_b_text = entities_b_dict.get(rel_b['from'], {'text': ''})['text']
                to_b_text = entities_b_dict.get(rel_b['to'], {'text': ''})['text']
                csvwriter.writerow([
                    from_a_text, rel_a['label'], to_a_text,
                    from_b_text, rel_b['label'], to_b_text,
                    f"{avg_score:.2f}"
                ])
            csvwriter.writerow([])
            
            # Extra Relations
            csvwriter.writerow(['Extra Relations in Generated Graph'])
            csvwriter.writerow(['From', 'Label', 'To'])
            for rel in analysis['extra_relations']:
                from_text = entities_b_dict.get(rel['from'], {'text': ''})['text']
                to_text = entities_b_dict.get(rel['to'], {'text': ''})['text']
                csvwriter.writerow([from_text, rel['label'], to_text])
            csvwriter.writerow([])
            
            # Missing Relations
            csvwriter.writerow(['Missing Relations from Generated Graph'])
            csvwriter.writerow(['From', 'Label', 'To'])
            for rel in analysis['missing_relations']:
                from_text = entities_a_dict.get(rel['from'], {'text': ''})['text']
                to_text = entities_a_dict.get(rel['to'], {'text': ''})['text']
                csvwriter.writerow([from_text, rel['label'], to_text])
            csvwriter.writerow([])
            
            csvwriter.writerow([])
        
    print("Comparison complete. Results have been saved to 'comparison_results.csv'.")

if __name__ == "__main__":
    main()
