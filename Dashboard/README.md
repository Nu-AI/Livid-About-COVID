
Given below is the list of instructions to run the dashboard file

---

+ Copy the entire dashboard directory to the working directory including the assets folder.

+ Install the requirements of the dependencies from the text file. 
  ```python
  pip install -r requirements.txt
  ```
+ Generate the state level data from the `get_state_data.py` to create the `formatted_all_data.csv`.
    * This generated data will be used to populate the dashboard and is necessary to be executed before 
    running the dashboard.

+ Run the `texas_dashboard.py` file using 
  ```python
  python texas_dashboard.py
  ```
+ This will open a port on localhost and display the address `http://127.0.0.1:8050/` where you can click and access the dashboard.

+ After making changes, rerun the code and keep accessing the port and it is recommended to close the tab before rerunning the code.

To install the systemd service, symlink the timer and service files to the right place
```shell
sudo ln -s "$PWD/update_predictions.service" /etc/systemd/system/
sudo ln -s "$PWD/update_predictions.timer" /etc/systemd/system/
sudo systemctl enable --now update_predictions.timer
```
=======
+ Site will be publicly available at [livid-about-covid19.nuai.utsa.edu/](http://livid-about-covid19.nuai.utsa.edu/)
