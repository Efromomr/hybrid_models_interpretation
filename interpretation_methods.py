#interpretation_methods.py

def lime(instance, prediction, tokens, mask, attention, hidden_states):
		""" This function represents the lime explainer. From the arguments it uses only the tokens
		Args:
			tokens: The tokenized instance
		Return:
			interpretations: It returns the extracted interpretations per label
		"""
		def predictor(texts):
			""" This function is a surrogate predict function. We use this one because the trainer/model does not provide the output in the shape lime needs it.
			Args:
				texts: the neighbours of the examined instance
			Return:
				all_probabilities: the predictions for the neighbours
			"""
			all_probabilities = []
			splits = np.array_split(texts, 50)
			for split in splits:
				split_labels = [0] * len(split)
				tokens = tokenizer(split[0], max_length=512, padding='max_length', truncation=True).input_ids[1:-1]


				logits = check_model(torch.tensor(tokens).unsqueeze(0))[0]
				probabilities = torch.nn.Softmax()(logits).detach().numpy()
				all_probabilities.append(probabilities)
			return np.array(all_probabilities)

		temp_instance = instance#fix_instance(tokens)

		interpretations = []
		for label in range(len(label_names)):
			exp = lime_explainer.explain_instance(temp_instance, predictor, num_features=512,
													   num_samples=50, labels=(label,))
			explanation_dict = {x[0]: x[1] for x in exp.as_list(label=label)}
			scores = []
			for tok in tokens[1:]:
				if tok == '[SEP]':
					break
				if tok not in string.punctuation:
					scores.append(explanation_dict[tok])
				else:
					scores.append(0)
			interpretations.append(scores)
		return interpretations
		
def my_attention(instance, prediction, tokens, mask, attention_i, hidden_states):
		""" This function represents the lime explainer. From the arguments it uses only the tokens and attention_i
		Args:
			tokens: The tokenized instance
			attention_i: The attention matrices as calculated for the examined instance
		Return:
			interpretations: It returns the extracted interpretations per label
		"""
		layers = attn_cfg['layers'] # Mean, Multi, Sum, First, Last
		heads = attn_cfg['heads'] # Mean, Sum, First, Last
		matrix = attn_cfg['matrix'] # From, To, MeanColumns, MeanRows, MaxColumns, MaxRows
		selection = attn_cfg['selection'] #True: select layers per head, False: do not

		attention = attention_i.detach().numpy()#.copy()
		if not selection:

			if heads == 'Mean':
					attention = attention.mean(axis=1)
			elif heads == 'Sum':
					attention = attention.sum(axis=1)
			elif type(heads) == type(1):
				attention = attention[:,heads,:,:]

			if layers == 'Mean':
				attention = attention.mean(axis=0)
			elif layers == 'Sum':
				attention = attention.sum(axis=0)
			elif layers == 'Multi':
				joint_attention = attention[0]
				for i in range(1, len(attention)):
					joint_attention = np.matmul(attention[i],joint_attention)
				attention = joint_attention
			elif type(layers) == type(1):
				attention = attention[layers]

			if matrix == 'From':
				attention = attention[0]
			elif matrix == 'To':
				attention = attention[:,0]
			elif matrix == 'MeanColumns':
				attention = attention.mean(axis=0)
			elif matrix == 'MeanRows':
				attention = attention.mean(axis=1)
			elif matrix == 'MaxColumns':
				attention = attention.max(axis=0)
			elif matrix == 'MaxRows':
				attention = attention.max(axis=1)
		else:
			importance_attention_matrices = []
			for i in range(len(attention)):
				att_heads = []
				for j in range(len(attention[0])):
					mm = attention_i[i][j][1:-1,1:-1].max()
					if mm > 0.5:
						indi = 0
						indj = 0
						for k in np.argmax(attention_i[i][j][1:-1,1:-1],axis=0):
							if mm in attention_i[i][j][1:-1,1:-1][k]:
								indi = k
								indj = np.argmax(attention_i[i][j][1:-1,1:-1][k])
						if abs(indi-indj) != 0:
							att_heads.append(attention_i[i][j])
				if heads == 'Mean' and len(att_heads) > 0:
						importance_attention_matrices.append(np.array(att_heads).mean(axis=0))
				elif heads == 'Sum' and len(att_heads) > 0:
						importance_attention_matrices.append(np.array(att_heads).sum(axis=0))
			importance_attention_matrices = np.array(importance_attention_matrices)

			if layers == 'Mean':
				attention = importance_attention_matrices.mean(axis=0)
			elif layers == 'Sum':
				attention = importance_attention_matrices.sum(axis=0)
			elif layers == 'Multi':
				attention = importance_attention_matrices[0]
				for i in range(1,len(importance_attention_matrices)):
					attention = np.matmul(attention,importance_attention_matrices[i])
			attention=attention[0]
		interpretations = []
		for label in range(len(label_names)):
			interpretations.append(attention[1:-1])
		#if self.sentence_level:
			#interpretations = self.convert_to_sentence(tokens, interpretations)
		return interpretations