#!pip install bert-tensorflow
# or
#git clone https://github.com/google-research/bert.git
from csv import DictReader
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from tensorflow import keras
import os
import re

#labels i.e. Discuss, Agree, Disagree, Unrelated
LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
HEADLINE_COLUMN = 'headline'
BODY_COLUMN = 'body'
LABEL_COLUMN = 'stance'
# label_list is the list of labels 0, 1 ,2,3
label_list = [0,1,2,3]  
# Set the output directory for saving model file
OUTPUT_DIR = 'Running_Model'
#Whether or not to clear/delete the directory and create a new one
DO_DELETE = True 
#uncased all lowercase version of BERT
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
# Compute train and warmup steps from batch size
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3.0
# Warmup is a period of time where learning rate is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 500
SAVE_SUMMARY_STEPS = 100
#set sequence length to be 128 tokens long.
MAX_SEQ_LENGTH = 128

################################Class DataSet starts##############################################################
class DataSet():
    def __init__(self, name="train", path=""): #FakeNewsChallenge, drive/FakeNewsChallenge
        self.path = path

        print("Reading dataset")
        bodies = name+"_bodies.csv"
        stances = name+"_stances.csv"

        self.stances = self.read(stances)
        articles = self.read(bodies)
        self.articles = dict()

        #make the body ID an integer value
        for s in self.stances:
            s['Body ID'] = int(s['Body ID'])

        #copy all bodies into a dictionary
        for article in articles:
            self.articles[int(article['Body ID'])] = article['articleBody']

        print("Total stances: " + str(len(self.stances)))
        print("Total bodies: " + str(len(self.articles)))



    def read(self,filename):
        rows = []
        with open( filename, "r", encoding='utf-8') as table:  #self.path + "/" +
            r = DictReader(table)

            for line in r:
                rows.append(line)
        return rows
##############################Class DataSet ends################################################################      

def generate_dataframe(stances,dataset):
	
	'''creates three lists label,Headline,Body ID
        and converts to Dataframe and returns dataframe
        Parameters:
            stances  has stance, bodyid and Headline
            dataset : DataSet class object has articles corresponding to bodyIDs
        Returns:
            dataframe:Returning  dataframe'''
	
	h, b, y = [],[],[]

	for stance in stances:
		y.append(LABELS.index(stance['Stance']))
		h.append(stance['Headline'])
		b.append(dataset.articles[stance['Body ID']])
	df = pd.DataFrame(data = {'stance':y,'headline':h,'body':b })
	return df
	
def create_tokenizer_from_hub_module():
	"""Getting the vocab file 
	and casing information
	from the Hub module."""
	with tf.Graph().as_default():
		bert_module = hub.Module(BERT_MODEL_HUB)
		tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
		with tf.Session() as sess:
			vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])
      
	return bert.tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

########################################Creating a model###########################################
def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,
                 num_labels):
	"""will create a classification model."""

	bert_module = hub.Module(BERT_MODEL_HUB,trainable=True)
  
	bert_inputs = dict(input_ids=input_ids,input_mask=input_mask,segment_ids=segment_ids)
	bert_outputs = bert_module(inputs=bert_inputs,signature="tokens",as_dict=True)

	# Use "pooled_output" for classification tasks on an entire sentence.
	output_layer = bert_outputs["pooled_output"]

	hidden_size = output_layer.shape[-1].value

	# Create our own layer to tune for stance data.
	output_weights = tf.get_variable("output_weights", [num_labels, hidden_size],initializer=tf.truncated_normal_initializer(stddev=0.02))

	output_bias = tf.get_variable("output_bias", [num_labels], initializer=tf.zeros_initializer())

	with tf.variable_scope("loss"):

		# Dropout helps prevent overfitting
		output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

		logits = tf.matmul(output_layer, output_weights, transpose_b=True)
		logits = tf.nn.bias_add(logits, output_bias)
		log_probs = tf.nn.log_softmax(logits, axis=-1)

		# Convert labels into one-hot encoding
		one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

		predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
		# If we're predicting, we want predicted labels and the probabiltiies.
		if is_predicting:
			return (predicted_labels, log_probs)

		# If we're train/eval, compute loss between predicted and actual label
		per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
		loss = tf.reduce_mean(per_example_loss)
		return (loss, predicted_labels, log_probs)

# model_fn_builder actually creates the model function
# using the passed parameters for num_labels, learning_rate, etc.
def model_fn_builder(num_labels, learning_rate, num_train_steps, num_warmup_steps):
	"""Returns `model_fn` closure for TPUEstimator."""
	def model_fn(features, labels, mode, params):  
		"""The `model_fn` for TPUEstimator."""

		input_ids = features["input_ids"]
		input_mask = features["input_mask"]
		segment_ids = features["segment_ids"]
		label_ids = features["label_ids"]

		is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)
		# TRAIN and EVAL
		if not is_predicting:

			(loss, predicted_labels, log_probs) = create_model(is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

			train_op = bert.optimization.create_optimizer(loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

			# Calculate evaluation metrics i.e. accuracy. 
			def metric_fn(label_ids, predicted_labels):
				accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
				return {"eval_accuracy": accuracy}

			eval_metrics = metric_fn(label_ids, predicted_labels)

			if mode == tf.estimator.ModeKeys.TRAIN:
				return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)
			else:
				return tf.estimator.EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=eval_metrics)
		else:
			(predicted_labels, log_probs) = create_model(is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

			predictions = {
			'probabilities': log_probs,
			'labels': predicted_labels
		  }
			return tf.estimator.EstimatorSpec(mode, predictions=predictions)

	# Return the actual model function in the closure
	return model_fn	

if DO_DELETE:
	try:
		tf.gfile.DeleteRecursively(OUTPUT_DIR)
	except:
		pass # if the directory didn't exist, it doesn't matter 
        
tf.gfile.MakeDirs(OUTPUT_DIR)
print('Model output directory: {} '.format(OUTPUT_DIR))

#Load the training dataset 
d = DataSet()
#Load the competition dataset 
competition_dataset = DataSet("competition_test")

#get the dataframe with features i.e. body, headline   
train = generate_dataframe(d.stances, d)
test = generate_dataframe(competition_dataset.stances, competition_dataset)


####################################Data Preprocessing#############################################

# Use the InputExample class from BERT's run_classifier code to create examples from the data
train_InputExamples = train.apply(lambda x: bert.run_classifier.InputExample(guid=None, # Globally unique ID for bookkeeping, not used here
                                                                   text_a = x[HEADLINE_COLUMN], 
                                                                   text_b = x[BODY_COLUMN], 
                                                                   label = x[LABEL_COLUMN]), axis = 1)

test_InputExamples = test.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                   text_a = x[HEADLINE_COLUMN], 
                                                                   text_b = x[BODY_COLUMN], 
                                                                   label = x[LABEL_COLUMN]), axis = 1)


#creating toeknizer
tokenizer = create_tokenizer_from_hub_module()

# Convert our train and test features to InputFeatures that BERT understands.
train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)

# Compute train and warmup steps from batch size
num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

# Specify output directory and number of checkpoint steps to save
run_config = tf.estimator.RunConfig(
    model_dir=OUTPUT_DIR,
    save_summary_steps=SAVE_SUMMARY_STEPS,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

model_fn = model_fn_builder(
  num_labels=len(label_list),
  learning_rate=LEARNING_RATE,
  num_train_steps=num_train_steps,
  num_warmup_steps=num_warmup_steps)

estimator = tf.estimator.Estimator(
  model_fn=model_fn,
  config=run_config,
  params={"batch_size": BATCH_SIZE})

# Create an input function for training. drop_remainder = True for using TPUs.
train_input_fn = bert.run_classifier.input_fn_builder(
    features=train_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=True,
    drop_remainder=False)

print('Beginning Training!')
#we feed our estimator with the train input function and the features and labels created at the beginning.
current_time = datetime.now()
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
print("Training took time ", datetime.now() - current_time)
test_input_fn = run_classifier.input_fn_builder(
    features=test_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=False)

# we pass the evaluation input function with test features and labels, and the batch size.
estimator.evaluate(input_fn=test_input_fn, steps=None)

#estimator predictions on test/competition dataset
predictions = estimator.predict(test_input_fn)

test_prediction_list = []
for pred_dict in predictions :  
    class_id = pred_dict['labels']
    test_prediction_list.append(class_id)

#saving the test predictions in Dataframe
competition_result = pd.DataFrame() 
competition_result['Predicted_stance'] = test_prediction_list  
competition_result['Predicted_stance_Actual'] = competition_result['Predicted_stance'].map({0: 'agree', 1:'disagree',2:'discuss',3:'unrelated' })
competition_result.to_csv("Competition_result.csv", index = False)