** Running the python script **
The script is named A2LSH.py
The script is currently made to run for the DNA dataset and hence takes a tab seperated tsv file as input with rows as documnets (sequences in the DNA dataset).
Please ensure you have all the required packages installed before you run the script.

1. Make sure that 'apples.jpg' is in the same directory as the python script.
2. Run the A2LSH.py from terminal or any IDE.
3. You will be greeted by a GUI.
4. Enter shingle size, number of permutations for Minhashing and number of bands for LSH.
5. Enter the path for the human_data.txt file.
6. Let it run. The GUI will be not responding while the LSH runs in the background since this is a single threaded application.
7. After completion, GUI will display the time taken.
8. Enter query.
9. If query is seq from dataset, results are displayed instantly.
10. If query is random seq, wait for processing (GUI will be not responding) and then results will be displayed.