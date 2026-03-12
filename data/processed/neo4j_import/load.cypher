LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
MERGE (n:Entity {node_id: row.`node_id:ID`})
SET n.label = row.label,
    n.doc_id = row.doc_id,
    n.start_page = CASE WHEN row.`start_page:int` = '' THEN null ELSE toInteger(row.`start_page:int`) END,
    n.end_page = CASE WHEN row.`end_page:int` = '' THEN null ELSE toInteger(row.`end_page:int`) END,
    n.display_name = row.display_name,
    n.json_properties = row.json_properties;

LOAD CSV WITH HEADERS FROM 'file:///edges.csv' AS row
MATCH (s:Entity {node_id: row.`:START_ID`})
MATCH (t:Entity {node_id: row.`:END_ID`})
CALL apoc.create.relationship(
  s,
  row.`:TYPE`,
  {
    edge_id: row.edge_id,
    doc_id: row.doc_id,
    start_page: CASE WHEN row.`start_page:int` = '' THEN null ELSE toInteger(row.`start_page:int`) END,
    end_page: CASE WHEN row.`end_page:int` = '' THEN null ELSE toInteger(row.`end_page:int`) END,
    json_properties: row.json_properties
  },
  t
) YIELD rel
RETURN count(rel);
