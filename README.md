# **xPred**
## The interactive tool for analizyng and predicting football matches
#### ⚠️ 
##### For a better visualization is strongly recommended using the *dark theme*. If you see light background (white one), please go on top-right corner<br>*burger button* --> *settings* --> choose *Dark* theme.

### **Introduction**
This application is developed assuming the scenario for supporting technical staff during the match preparation, supporting them for extracting knowledge based on previous played matches. Used data are related to **Serie A 2021/2022**.


### **Structure**
We have basically 4 sections:
- The first section contains a collapsable element that allows you to choose the round and the match to be analyzed
- The second section displays team logos and charts for drawing the *Indice di Pericolosità Offensiva (IPO)* produced over previous matches by the home and away teams. The background gray chart depicts the IPO produced against the considered team. Below this IPO chart, there is a bar chart that indicates if the considered (<span style="color:steelblue">home</span>/<span style="color:lightgreen">away</span>) team won against the opponent (3-units colored bar), draws (1-unit colored bar) or lost (3-unit gray bar)
- The third sections is the core part of the whole application. Here, all metrics (normalized per 90 mins) are split by their cluster (Offensive, Defensive, Passing and Physical). In the first part of this section, each metric has a subchart that shows the top 5 players with highest values in that metric. You can use the available range filter in order to filter-in or filter-out players from the analysis, taking into account the amount of mins played until the selected round. In the last part of that section, a Pizza chart (on left side) and pitch charts (on right side) expose metrics of the considered team. In Pizza chart, metrics are evaluated calculating the rank of considered teams wrt other teams of Serie A. On pitch charts, for the sake of visualization, only events related to last 5 matches are shown.
- Finally, in last section, We have an experimental section in which We try to retrieve the probability of scoring goals by the two teams. These probabilities are shown using a matrix heatmap. In order to extract these probabilities, goals of match are modeled as discrete events using the **Poisson** distribution. The *lambda* factor of the distribution is evaluated considering an average of expected metrics, such as: *Expected Goals (xG)*, and *Expected Assist (xA)* produced home and away during the last five matches by every team. In order to predict the probabities, each match is simulated 20000 (*twenty thousand*) times using the MonteCarlo Simulation.
*This is just a configuration, We have a thousands of possibilities combining the different metrics. The codebase for generating this file is in another repository.*

### **Datasets**
This dashboard is based on four dataset based on **Serie A 2021/2022**
1. dati_completi_serieA_2021-22_partite.csv - Contains all data related to players who played in any match of Serie A 2021/2022
2. SICS_SerieA_2021-22_OffensiveIndexTimetable.csv - Contains the set of avents (in any match) that increase the value of **IPO**
3. SICS-SerieA_2021-22_eventi.csv - Contains all events in any match played in the previous season of Serie A
4. simulated_matches.csv - Custom dataset containing the output of Monte Carlo Simulation with probabilities related to any match

### **Credits**
This app is entirely developed by [Mario Fresilli](https://www.linkedin.com/in/mariofresilli/)<br>
Datasets are provided by [SICS](https://www.sics.it/) and [Soccerment](https://soccerment.com/)





<img src="https://media-exp1.licdn.com/dms/image/C4D03AQG3JZV5cOrmJA/profile-displayphoto-shrink_400_400/0/1659475596832?e=1675296000&v=beta&t=DHuhKaRUeSsOchbJHssjJ6G8ebZ4E5Y78co_TrVn74o" style="border-radius:50%; height: 150px;">

<br>

![Soccerment](https://soccerment.com/wp-content/uploads/2017/08/soccerment-scritta360.png)

**<span style='color:orange; font-size:96px;'>SICS<span>**



