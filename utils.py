import xml.etree.ElementTree as ET
import tiktoken 

class StackExchangeXMLDataset:

	def is_question(self,row):
		return row.attrib["PostTypeId"] == "1"

	def is_question_answered(self,row):
		if not self.is_question(row): return False
		return "AcceptedAnswerId" in row.attrib.keys()

	def get_row_by_id(self,idx):
		for row in self.root:
			if row.attrib["Id"] == idx:
				return row 
		return None

	def format_body(self,row):
		return row.attrib["Body"].replace("<p>","").replace("</p>","")

	def get_qa_pairs(self):
		qa_pairs = [(self.format_body(row),self.format_body(self.get_row_by_id(row.attrib["AcceptedAnswerId"]))) for row in self.root if self.is_question_answered(row)]
		return qa_pairs

	def __init__(self,filepath,enc_type="cl100k_base"):
		self.enc = tiktoken.get_encoding(enc_type)
		print(f"Using encoding with vocab size: {self.enc.n_vocab}")
		self.filepath = filepath 
		self.tree = ET.parse(filepath)
		self.root = self.tree.getroot()




if __name__ == "__main__":
	parser = StackExchangeXMLDataset("data/datasciencestackexchangeposts.xml")
	qa_pairs = parser.get_qa_pairs()
	print(qa_pairs)
	print(len(qa_pairs))

