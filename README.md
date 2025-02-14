# SignedGraphToolbox
A toolbox for signal processing, unsupervised learning and semi-supervised learning on signed graphs


# How to Launch simulations
## Make Result Directories
Run
```
python3 -m src.launcher.TV.sbm_sim -mk
```
to make the directories for all results if they do not exist

## Get List of Setting
Run
```
python3 -m src.launcher.TV.sbm_sim -n
```
to get an overview of available custom simulations. Each simulation ID contains several different configs that correspond to different settings for the graph.

## Single-Sim Run
Launch a single simulation with
```
python3 -m src.launcher.TV.sbm_sim <offset> <process_id> <sim_id>
```
where
 - process_id (modulo the number of configs) is the id of the config to run and also corresponds to the random seed of the simulation,
 - offset is added to the process_id (this can be utilized when running many indepentend simulation threads on a compute cluster),
 - and sim_id specifies, which setting is simulated

 ## Multi-Sim Run
Launch a range of simulations with
```
python3 -m src.launcher.TV.sbm_sim <offset> <process_id_start>-<process_id_stop> <sim_id>
```
In order to run several configs for one setting
