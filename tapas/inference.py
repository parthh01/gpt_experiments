from transformers import AutoTokenizer, TapasForQuestionAnswering
import pandas as pd


def main():
	tokenizer = AutoTokenizer.from_pretrained("google/tapas-base-finetuned-wtq")
	model = TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq")

	data = {
    		"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
    		"Age": ["56", "45", "59"],
    		"Number of movies": ["87", "53", "69"],
	}
	table = pd.DataFrame.from_dict(data)
	queries = ["How many movies has George Clooney played in?", "How old is Brad Pitt?"]

	inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="pt")
	outputs = model(**inputs)

	logits = outputs.logits
	logits_aggregation = outputs.logits_aggregation
	print(dir(logits))


if __name__ == "__main__":
	main()
