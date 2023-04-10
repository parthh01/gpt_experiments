import xml.etree.ElementTree as ET
import tiktoken 
import torch

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
        self.qa_pairs = qa_pairs
        return None
    
    def format_qa_pairs_for_model(self,qa_pairs):
        return [q + "<|endofprompt|>" + a + "<|endoftext|>" for q,a in qa_pairs] 
    
    def translate(self,data,enc=True):
        return [self.enc.encode(p,allowed_special={*self.TOKEN_MAP.keys()}) if enc else self.enc.decode(p) for p in data] 
    
    def prepare_data_for_model(self,qa_pairs):
        formatted_rows = self.format_qa_pairs_for_model(qa_pairs)
        translated_rows = self.translate(formatted_rows)
        return translated_rows
    
    def get_batch(self,X,batch_size):
        idxs = torch.randint(len(X),(batch_size,))
        batch = [X[i] for i in idxs]
        context_idx = [p.index(self.TOKEN_MAP["<|endofprompt|>"]) for p in batch]
        target_idx = [len(p) for p in batch]
        batch_x = torch.stack([torch.tensor(x[:t-1],dtype=torch.long) for x,t in zip(batch,target_idx)])
        batch_y = torch.stack([torch.tensor(x[c+1:],dtype=torch.long) for x,c in zip(batch,context_idx)])
        return batch_x,batch_y
        
        
        
        
        
        
    
    def __init__(self,filepath,enc_type="r50k_base",train_val_split=0.8,SPECIAL_TOKENS = ['<|endofprompt|>','<|endoftext|>']):
        torch.manual_seed(7)
        custom_enc = tiktoken.get_encoding(enc_type)
        self.enc = tiktoken.Encoding(
            # If you're changing the set of special tokens, make sure to use a different name
            # It should be clear from the name what behaviour to expect.
            name=enc_type,
            pat_str=custom_enc._pat_str,
            mergeable_ranks=custom_enc._mergeable_ranks,
            special_tokens={
                **custom_enc._special_tokens
            }
        )
        self.TOKEN_MAP = {token: self.enc.encode(token,allowed_special={*SPECIAL_TOKENS})[0] for token in SPECIAL_TOKENS}
        print(f"Using encoding with vocab size: {self.enc.n_vocab}")
        self.filepath = filepath 
        self.tree = ET.parse(filepath)
        self.root = self.tree.getroot()
        self.train_val_split=train_val_split
        self.get_qa_pairs()
        self.train_data = self.qa_pairs[:int(len(self.qa_pairs)*train_val_split)]
        self.val_data = self.qa_pairs[int(len(self.qa_pairs)*train_val_split):]
        




if __name__ == "__main__":
    parser = StackExchangeXMLDataset("data/datasciencestackexchangeposts.xml")
    qa_pairs = parser.get_qa_pairs()
    print(qa_pairs)
    print(len(qa_pairs))

