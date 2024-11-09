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
    entities = {}
    for node in nodes_json:
        label = node.get('label', '')
        label_match = re.match(r'^(.*?)\s+<(.+?)>$', label)
        if label_match:
            text = label_match.group(1).strip()
            entity_type = label_match.group(2).strip()
            # Normalize text: lowercase and strip
            text_normalized = text.lower().strip()
            entity_type_normalized = entity_type.lower().strip()
            # Use normalized text and type
            entities[node['id']] = {
                'text': text_normalized,
                'type': entity_type_normalized,
                'id': node['id']
            }
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

def compute_similarity(text1, text2):
    # Normalize texts: lowercase and strip
    text1 = text1.lower().strip()
    text2 = text2.lower().strip()
    vectorizer = TfidfVectorizer().fit([text1, text2])
    tfidf = vectorizer.transform([text1, text2])
    cosine_sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return cosine_sim

def compare_entities(entities_a, entities_b, threshold=0.7):
    matched = []
    extra = entities_b.copy()
    missing = entities_a.copy()
    used_entities_b = set()  # Keep track of matched entities in entities_b
    
    for id_a, entity_a in entities_a.items():
        best_match_id = None
        best_score = 0
        for id_b, entity_b in entities_b.items():
            if id_b in used_entities_b:
                continue  # Skip entities already matched
            if entity_a['type'] == entity_b['type']:
                sim_score = compute_similarity(entity_a['text'], entity_b['text'])
                if sim_score > best_score:
                    best_score = sim_score
                    best_match_id = id_b
        if best_score >= threshold and best_match_id is not None:
            matched.append((entity_a, entities_b[best_match_id], best_score))
            used_entities_b.add(best_match_id)
            del extra[best_match_id]
            del missing[id_a]
    return matched, list(extra.values()), list(missing.values())

def compare_relations(relations_a, relations_b, entities_a, entities_b, threshold=0.7):
    matched = []
    extra = relations_b.copy()
    missing = relations_a.copy()
    used_relations_b = set()
    
    # Precompute entity text mapping
    entity_text_a = {entity['id']: entity['text'] for entity in entities_a.values()}
    entity_text_b = {entity['id']: entity['text'] for entity in entities_b.values()}
    
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
    if len(sys.argv) != 3:
        print("Usage: python Compare_Graphs.py <path_to_true_graph> <path_to_generated_graph>")
        sys.exit(1)
    
    true_graph_path = sys.argv[1]
    generated_graph_path = sys.argv[2]
    
    # Read HTML contents from files
    with open(true_graph_path, 'r', encoding='utf-8') as f:
        html_content_a = f.read()
    
    with open(generated_graph_path, 'r', encoding='utf-8') as f:
        html_content_b = f.read()
    
    # Extract data
    nodes_a, edges_a = extract_data(html_content_a)
    nodes_b, edges_b = extract_data(html_content_b)
    
    # Parse entities and relations
    entities_a = parse_entities(nodes_a)
    entities_b = parse_entities(nodes_b)
    relations_a = parse_relations(edges_a)
    relations_b = parse_relations(edges_b)
    
    # Compare entities using cosine similarity
    matched_entities, extra_entities, missing_entities = compare_entities(entities_a, entities_b)
    
    # Compare relations using cosine similarity
    matched_relations, extra_relations, missing_relations = compare_relations(relations_a, relations_b, entities_a, entities_b)
    
    # Count entities and relations
    total_entities_truth = len(entities_a)
    total_entities_generated = len(entities_b)
    total_relations_truth = len(relations_a)
    total_relations_generated = len(relations_b)
    
    # Output results to CSV
    with open('comparison_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        # Write header rows
        csvwriter.writerow(['Comparison Results'])
        csvwriter.writerow([])
        csvwriter.writerow(['Entity Comparison Summary'])
        csvwriter.writerow(['Total Entities in Truth Graph:', total_entities_truth])
        csvwriter.writerow(['Total Entities in Generated Graph:', total_entities_generated])
        csvwriter.writerow(['Matched Entities:', len(matched_entities)])
        csvwriter.writerow(['Extra Entities (in Generated Graph):', len(extra_entities)])
        csvwriter.writerow(['Missing Entities (from Generated Graph):', len(missing_entities)])
        csvwriter.writerow([])
        csvwriter.writerow(['Relation Comparison Summary'])
        csvwriter.writerow(['Total Relations in Truth Graph:', total_relations_truth])
        csvwriter.writerow(['Total Relations in Generated Graph:', total_relations_generated])
        csvwriter.writerow(['Matched Relations:', len(matched_relations)])
        csvwriter.writerow(['Extra Relations (in Generated Graph):', len(extra_relations)])
        csvwriter.writerow(['Missing Relations (from Generated Graph):', len(missing_relations)])
        csvwriter.writerow([])
        
        # Write Matched Entities
        csvwriter.writerow(['Matched Entities'])
        csvwriter.writerow(['Truth Entity Text', 'Generated Entity Text', 'Entity Type', 'Similarity Score'])
        for entity_a, entity_b, sim_score in matched_entities:
            csvwriter.writerow([entity_a['text'], entity_b['text'], entity_a['type'], f"{sim_score:.2f}"])
        csvwriter.writerow([])
        
        # Write Extra Entities
        csvwriter.writerow(['Extra Entities in Generated Graph'])
        csvwriter.writerow(['Entity Text', 'Entity Type'])
        for entity in extra_entities:
            csvwriter.writerow([entity['text'], entity['type']])
        csvwriter.writerow([])
        
        # Write Missing Entities
        csvwriter.writerow(['Missing Entities from Generated Graph'])
        csvwriter.writerow(['Entity Text', 'Entity Type'])
        for entity in missing_entities:
            csvwriter.writerow([entity['text'], entity['type']])
        csvwriter.writerow([])
        
        # Write Matched Relations
        csvwriter.writerow(['Matched Relations'])
        csvwriter.writerow(['Truth From', 'Truth Label', 'Truth To', 'Generated From', 'Generated Label', 'Generated To', 'Similarity Score'])
        for rel_a, rel_b, avg_score in matched_relations:
            csvwriter.writerow([
                entities_a[rel_a['from']]['text'], rel_a['label'], entities_a[rel_a['to']]['text'],
                entities_b[rel_b['from']]['text'], rel_b['label'], entities_b[rel_b['to']]['text'],
                f"{avg_score:.2f}"
            ])
        csvwriter.writerow([])
        
        # Write Extra Relations
        csvwriter.writerow(['Extra Relations in Generated Graph'])
        csvwriter.writerow(['From', 'Label', 'To'])
        for rel in extra_relations:
            from_text = entities_b.get(rel['from'], {'text': ''})['text']
            to_text = entities_b.get(rel['to'], {'text': ''})['text']
            csvwriter.writerow([from_text, rel['label'], to_text])
        csvwriter.writerow([])
        
        # Write Missing Relations
        csvwriter.writerow(['Missing Relations from Generated Graph'])
        csvwriter.writerow(['From', 'Label', 'To'])
        for rel in missing_relations:
            from_text = entities_a.get(rel['from'], {'text': ''})['text']
            to_text = entities_a.get(rel['to'], {'text': ''})['text']
            csvwriter.writerow([from_text, rel['label'], to_text])
        csvwriter.writerow([])
    
    print("Comparison complete. Results have been saved to 'comparison_results.csv'.")

if __name__ == "__main__":
    main()
