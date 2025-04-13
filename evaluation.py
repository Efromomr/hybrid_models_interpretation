#evaluation.py

def find_sign(weight):
		"""
			This function identifies the sign of a weight
		Args:
			weight: The weight of which the sign we want to identify
		Return:
			sign: The sign of the weights
		"""
		if weight < 0:
			sign = 'negative'
		elif weight > 0:
			sign = 'positive'
		else:
			sign = 'neutral'
		return sign

def apply_activation(pred1, pred2):
		"""
			This function identifies the sign of a weight
		Args:
			weight: The weight of which the sign we want to identify
		Return:
			predicted_labels: the predicted labels after the softmax or sigmoid functions
		"""
		predicted_labels = torch.nn.Softmax()(torch.tensor([pred1,pred2]))
		return predicted_labels

def fix_instance(instance):
		"""
			This function takes an instance like "The tok ##en ##ized sentence" and transforms it to "The tokenized sentence"
		Args:
			instance: The input sequence to be fixed from tokenized to original
		Return:
			new_sentence[1:]: The original sentence
		"""
		return " ".join([tok.replace('##', '') for tok in instance])


label_names = [0, 1]

def faithfulness(interpretation, tweaked_interpretation, instance, prediction, tokens, hidden_states, t_hidden_states, rationales, tokenized=False):
		"""
			This function evaluates an interpretation uzing the faithdulness score (F) metric
		Args:
			interpretation: The interpretations of the instance's prediction
			tweaked_interpretation: The interpretations of the tweaked instance's prediction (useful for robustness)
			instance: The input sequence to be fixed from tokenized to original
			prediction: The prediction regarding the input sequence
			tokens: The input sequence but tokenized
			hidden_states: The hidden states reagrding that instance extracted by the model
			t_hidden_states: The hidden states reagrding the tweaked instance extracted by the model (useful for robustness)
			rationales: The rationales (ground truth explanations - useful for auprc)
		Return:
			avg_diff: The average faithfulness score for each interpretation per label
		"""
		avg_diff = []
		predicted_labels = torch.nn.Softmax()(prediction[0])

		for label in range(len(label_names)):

			if predicted_labels[label]>=0.5:

				#print('For label:',self.label_names[label])
				absmax_index = np.argmax(interpretation[label])
				sign = find_sign(interpretation[label][absmax_index])
				if sign == 'negative':
					absmax_index = np.argmax([abs(i) for i in interpretation[label]])
					sign = find_sign(interpretation[label][absmax_index])
				#print('Argmax:',interpretation[label][absmax_index],'token:',temp_tokens[absmax_index+1])

				
				if not tokenized:
					temp_tokens = tokens.copy()
					temp_tokens[absmax_index+1] = '[UNK]'
					temp_instance = torch.tensor(tokenizer.encode(temp_tokens, is_split_into_words=True)[1:-1]).unsqueeze(0)
				else:
					temp_tokens = tokens
					temp_tokens[0][absmax_index+1] = 100
					temp_instance = temp_tokens
				temp_prediction = check_model(temp_instance)
				preds = [predicted_labels, torch.nn.Softmax()(temp_prediction[0])]
				if sign == 'positive':
					diff = preds[0][label] - preds[1][label]
				elif sign == 'negative':
					diff = preds[1][label] -  preds[0][label]
				else: #neutral
					diff = (-1)*abs(preds[1][label] -  preds[0][label]) #Penalty
				avg_diff.append(diff)
			else:
				avg_diff.append([])
		return avg_diff

def truthfulness(interpretation, tweaked_interpretation, instance, prediction, tokens, hidden_states, t_hidden_states, rationales):
		"""
			This function evaluates an interpretation uzing the truthfulness metric
		Args:
			interpretation: The interpretations of the instance's prediction
			tweaked_interpretation: The interpretations of the tweaked instance's prediction (useful for robustness)
			instance: The input sequence to be fixed from tokenized to original
			prediction: The prediction regarding the input sequence
			tokens: The input sequence but tokenized
			hidden_states: The hidden states reagrding that instance extracted by the model
			t_hidden_states: The hidden states reagrding the tweaked instance extracted by the model (useful for robustness)
			rationales: The rationales (ground truth explanations - useful for auprc)
		Return:
			avg_diff: The average truthfulness score for each interpretation per label
		"""
		avg_diff = []
		predicted_labels = torch.nn.Softmax()(prediction[0])

		for label in range(len(label_names)):

			if predicted_labels[label]>=0.5:
				truthful = 0
				my_range = len(tokens)-2
				#print('For label:',self.label_names[label])
				for token in range(0, my_range):

					temp_tokens = tokens.copy()

					temp_tokens[token] = ''
					temp_instance = tokenizer.encode(temp_tokens)
					temp_prediction = check_model(temp_instance)

					sign = 	find_sign(interpretation[label][token])
					#print('Token:',tokens[token+1],'Sign:',sign,'Weight:',interpretation[label][token])
					preds = [predicted_labels, torch.nn.Softmax()(temp_prediction[0])]
					if sign == 'positive':
						if preds[0][label] - preds[1][label] > 0:
							truthful += 1
					elif sign == 'negative':
						if  preds[1][label] -  preds[0][label] > 0:
							truthful +=1
					else:
						if preds[1][label] ==  preds[0][label]:
							truthful +=1
					#print('Prevs:',preds[0][label],'Latter:',preds[1][label])
				avg_diff.append(truthful/my_range)
			else:
				avg_diff.append(np.average([]))
		return avg_diff

def auprc(interpretation, tweaked_interpretation, instance, prediction, tokens, hidden_states, t_hidden_states, rationales):
		"""
			This function evaluates an interpretation uzing the AUPRC metric
		Args:
			interpretation: The interpretations of the instance's prediction
			tweaked_interpretation: The interpretations of the tweaked instance's prediction (useful for robustness)
			instance: The input sequence to be fixed from tokenized to original
			prediction: The prediction regarding the input sequence
			tokens: The input sequence but tokenized
			hidden_states: The hidden states reagrding that instance extracted by the model
			t_hidden_states: The hidden states reagrding the tweaked instance extracted by the model (useful for robustness)
			rationales: The rationales (ground truth explanations - useful for auprc)
		Return:
			avg_diff: The average AUPRC score for each interpretation per label
		"""
		aucs = []
		predicted_labels = apply_activation(prediction, prediction)[0]

		for label in range(len(label_names)):
			if predicted_labels[label]>=0.5:
				label_auc = []
				if rationales[label] != 0 and sum(rationales[label]) != 0:
					precision, recall, _ = precision_recall_curve(rationales[label],interpretation[label])
					label_auc.append(auc(recall, precision))
				aucs.append(np.average(label_auc))
			else:
				aucs.append(np.average([]))
		return aucs