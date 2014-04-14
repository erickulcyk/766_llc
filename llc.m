function [predicted_label, accuracy, prob_estimates] = llc (K, data_dir, dictionary_file, train_file_names, train_label_vector, test_file_names, test_label_vector)

load(dictionary_file, 'dictionary');
train_c_out = GetLLCFeatures(K, dictionary, data_dir, train_file_names);
model = train(train_label_vector, train_c_out);

test_c_out = GetLLCFeatures(K, dictionary, data_dir, test_file_names);
[predicted_label, accuracy, prob_estimates] = predict(test_label_vector, test_c_out, model);
