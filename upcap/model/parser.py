import spacy
from spacy.tokens import Doc

class TextParser:
	
	def __init__(self):
		self.nlp = spacy.load("en_core_web_sm")
		self.expletives = {'there'}
	
	def __call__(self, text: str) -> list[str]:
		doc: Doc = self.nlp(text)
		concepts = [text]
		
		for chunk in doc.noun_chunks:
			if chunk.root.dep_ == 'expl' or chunk.text.lower() in self.expletives:
				continue
			
			head = chunk.root.head
			concept_text = chunk.text
			
			if head.pos_ == 'ADP' and head.i == chunk.start - 1:
				concept_text = f"{head.text} {concept_text}"
			
			concepts.append(concept_text)
		
		return concepts

if __name__ == '__main__':
	text_parser = TextParser()
	result = text_parser('there is a mie in japan.')
	print(result)