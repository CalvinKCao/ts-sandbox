# Metrix

<span id="Aperçu"></span>
= Summary =

[thumb|900px|center](File:Aperçu_de_la_page_d'accueil_du_portail.png.md)

The Metrix portal is a website for Alliance users. It collects information on compute nodes and management servers, to interactively generate data that allow users to track their resource usage (CPU, GPU, memory, filesystem) in real time.

{| class="wikitable"
|-
| Rorqual
| <https://metrix.rorqual.alliancecan.ca>(https://metrix.rorqual.alliancecan.ca)
|-
| Narval
| <https://portail.narval.calculquebec.ca>(https://portail.narval.calculquebec.ca)
|-
| Nibi
| <https://portal.nibi.sharcnet.ca>(https://portal.nibi.sharcnet.ca)
|}

 
<b>Filersystem performance</b>

Here you have the graphs for bandwidth and metadata operations, along with viewing options (last week, last day and last hour).

<b>Login nodes</b>

CPU, memory, system load, and network usage statistics are presented, with viewing options (last week, last day, and last hour).

<b>Scheduling</b>

This tab shows statistics for the cluster's allocated cores and GPUs, with viewing options (last week, last day, and last hour).

<b>Scientific software</b>

These graphs show the software with CPU cores and GPUs that are more frequently used.

<b>Data transfer nodes</b>

Bandwidth statistics for data transfer nodes are presented under this tab.

= User summary =
Under this tab, you find your quotas for various filesystems, followed by your 10 last tasks. You can select a task by its number to see the details. Also, by clicking on <span style="color:#0000FF">(More Details)</span>, you are redirected to the *Task statistics* tab, where all your tasks are listed.
[thumb|900px|center](File:Home.png.md)
[thumb|900px|center](File:Scratch.png.md)
[thumb|900px|center](File:Project.png.md)
[900px|thumb|center](File:Portail_utilisateur_10_dernières_tâches.png.md)

= Task statistics =
The first block shows your current usage (CPU core, memory, and GPUs). These statistics represent the average resources used by all currently running tasks. You can easily compare the resources allocated to you with those you actually use.
[thumb|900px|center](File:Utilisation_en_cours.png.md)

You then have access to an average of the last few days, presented in the form of a graph.
[thumb|900px|center](File:Coeur_CPU_Mémoire.png.md)

You then have a representation of your activity on the filesystems. On the left, the graph shows the number of disk write commands you have performed. (*input/output operations per second (IOPS)*) On the right, you see the amount of data transferred to the servers over a given period. (Bandwidth)
[thumb|900px|center](File:Système_de_fichier.png.md)

The next section shows all the tasks you have already started, which are currently running or pending. In the top left corner, you can filter tasks by their status (OOM, completed, running, etc.). In the top right corner, you can search by job ID or by name. Finally, in the bottom right corner, there is an option to quickly navigate between pages by performing multiple jumps. 
[thumb|900px|center](File:Vos_tâches_top-2.png.md)

[thumb|900px|center](File:Vos_tâches_bottom-2.png.md)

## CPU task page
At the top, you see the task name, its number, your username, and the status. Details of your submission script are displayed by clicking on <span style="color:white; background-color:blue">Voir le script de la tâche</span>. If the task was launched in interactive mode, the submission script will not be available.
[thumb|900px|center](File:Détails_sur_la_tâche-2.png.md)

<div lang="fr" dir="ltr" class="mw-content-ltr">
Le répertoire de travail et la commande de soumission sont accessibles en cliquant sur <span style="color:white; background-color:blue">Voir la commande de soumission</span>.
[thumb|900px|center](File:Commande_de_soumission-3.png.md)
</div>

<div lang="fr" dir="ltr" class="mw-content-ltr">
La prochaine section est dédiée aux informations de l'ordonnanceur. Vous pouvez accéder à la page de suivi de votre compte CPU en cliquant sur le numéro de votre compte. 
[thumb|900px|center](File:Information_ordonnanceur-2.png.md)
</div>

<div lang="fr" dir="ltr" class="mw-content-ltr">
Dans la section **Ressources** vous pouvez obtenir un aperçu initial de l'utilisation des ressources de votre tâche en comparant les colonnes **Alloués** et **Utilisés** pour les différents paramètres listés.
[thumb|900px|center](File:Ressources.png.md)
</div>

<div lang="fr" dir="ltr" class="mw-content-ltr">
Le graphique **CPU** vous permet de visualiser, dans le temps, des cœurs CPUs que vous avez demandés. À droite, vous pouvez sélectionner/désélectionner les différents cœurs selon vos besoins. Notez que pour des tâches très courtes, ce graphique n'est pas disponible.
[thumb|900px|center](File:Ressources_utilisées_détails-2.png.md)
</div>

<div lang="fr" dir="ltr" class="mw-content-ltr">
Le graphique **Mémoire** vous permet de visualiser, dans le temps, l'utilisation de la mémoire que vous avez demandée. 
[thumb|900px|center](File:Mémoire.png.md)
</div>

<div lang="fr" dir="ltr" class="mw-content-ltr">
Le graphique **Process and threads** vous permet d'observer différents paramètres liés aux processus et aux fils d'exécution. Idéalement, pour une tâche multifils (multithreading), l'addition du paramètre **Running threads** et **Sleeping threads** ne devrait pas dépasser de 2 fois le nombre de cœurs demandé. Cela dit, il est tout à fait normal d'avoir quelques processus en mode **dormant** (*Sleeping threads*) pour certain type de programmes (java, Matlab, logiciels commercial ou programmes complexes). Vous avez aussi en paramètre les applications du programme exécutées au fil du temps. 
[thumb|900px|center](File:Process_and_threads.png.md)
</div>


<div lang="fr" dir="ltr" class="mw-content-ltr">
Les graphiques suivants représentent l'utilisation du système de fichier pour la tâche en cours et non du nœud au complet. À gauche, une représentation du nombre d’opérations d’entrée/sortie par seconde (IOPS) est affichée. À droite, le graphique illustre le débit de transfert de données entre la tâche et le système de fichiers au fil du temps. Ce graphique permet d’identifier les périodes d’activité intense ou de faible utilisation du système de fichiers.
[thumb|900px|center](File:Système_de_fichier_-2.png.md)
</div>

Resource statistics for the entire node may be inaccurate if the node is shared by multiple users. The graph on the left shows the evolution of the bandwidth used by the task over time, in relation to software, licenses, etc. The graph on the right shows the evolution of the network bandwidth used by a task or a set of tasks via the Infiniband network, over time. We can observe periods of massive data transfer (e.g., reading/writing on a filesystem (Lustre), MPI communication between nodes).


<div lang="fr" dir="ltr" class="mw-content-ltr">
Le graphique de gauche illustre l’évolution du nombre d’opérations d’entrée/sortie par seconde (IOPS) effectuées sur le disque local au fil du temps. Celui de droite montre l’évolution de la bande passante utilisée sur le disque local au fil du temps, c’est-à-dire la quantité de données lues ou écrites par seconde.
[thumb|900px|center](File:IOPS,_bande_passante.png.md)
</div>

Use of local disk space
[thumb|900px|center](File:Espace_utilisé_sur_le_disque_local.png.md)

Capacity used
[thumb|900px|center](File:Puissance.png.md)

<div lang="fr" dir="ltr" class="mw-content-ltr">
## Page d'une tâche CPU (vecteur de tâches, *job array*)
</div>

<div lang="fr" dir="ltr" class="mw-content-ltr">
La page d'une tâche CPU dans un vecteur de tâches est identique à celle d'une tâche CPU régulière, à l'exception de la section *Other jobs in the array*. Le tableau liste les autres numéros de tâches faisant partie du même vecteur de tâches, ainsi que des informations sur leur statut, leur nom, leur heure de début et leur heure de fin.
</div>

[thumb|900px|center](File:CPU_job_array.png.md)

<div lang="fr" dir="ltr" class="mw-content-ltr">
## Page d'une tâche GPU
</div>

<div lang="fr" dir="ltr" class="mw-content-ltr">
En haut de page, vous avez le nom de la tâche, son numéro et votre nom d'utilisateur ainsi que le statut. Les détails de votre script de soumission s'affichent en cliquant sur <span style="color:white; background-color:blue">Voir le script de la tâche</span>. Si vous avez lancé une tâche interactive, le script de soumission n'est pas disponible.
[thumb|900px|center](File:Détail_de_la_tâche.png.md)
</div>

<div lang="fr" dir="ltr" class="mw-content-ltr">
Le répertoire et la commande de soumission sont accessibles en cliquant sur <span style="color:white; background-color:blue">Voir la commande de soumission</span>.
[thumb|900px|center](File:Commande_de_soumission-GPU.png.md)
</div>

<div lang="fr" dir="ltr" class="mw-content-ltr">
La section suivante est réservée aux informations de l'ordonnanceur. Vous pouvez accéder à la page de votre compte GPU en cliquant sur le numéro de votre compte. 
[thumb|900px|center](File:Information_ordonnanceur-GPU.png.md)
</div>

<div lang="fr" dir="ltr" class="mw-content-ltr">
Dans la section **Ressources** vous pouvez obtenir un premier aperçu de l'utilisation des ressources de votre tâche en comparant les colonnes **Alloués** et **Utilisés** pour les différents paramètres listés.
[thumb|900px|center](File:Ressources-GPU.png.md)
</div>

<div lang="fr" dir="ltr" class="mw-content-ltr">
Le graphique **CPU** vous permet de visualiser l'utilisation des cœurs CPUs demandés au fil du temps. À droite, vous pouvez sélectionner/désélectionner les différents cœurs selon vos besoins. Notez que pour des tâches très courtes, ce graphique n'est pas disponible.
[thumb|900px|center](File:CPU_ressources_utilisés_détails.png.md)
</div>

<div lang="fr" dir="ltr" class="mw-content-ltr">
Le graphique **Mémoire** vous permet de visualiser l'utilisation dans le temps de la mémoire que vous avez demandée pour les CPU. 
[thumb|900px|center](File:Mémoire-GPU.png.md)
</div>

<div lang="fr" dir="ltr" class="mw-content-ltr">
Le graphique **Process and threads** vous permet d'observer différents paramètres liés aux processus et aux fils d'exécution.
[thumb|900px|center](File:Processes_and_threads-GPU.png.md)
</div>

<div lang="fr" dir="ltr" class="mw-content-ltr">
Les graphiques suivants représentent l'utilisation du système de fichier pour la tâche en cours et non du nœud au complet. À gauche, une représentation du nombre d’opérations d’entrée/sortie par seconde (IOPS) est affichée. À droite, le graphique illustre le débit de transfert de données entre la tâche et le système de fichiers au fil du temps. Ce graphique permet d’identifier les périodes d’activité intense ou de faible utilisation du système de fichiers.
[thumb|900px|center](File:Systeme_de_fichiers-GPU.png.md)
</div>

<div lang="fr" dir="ltr" class="mw-content-ltr">
Le graphique GPU représente votre utilisation des GPU. Le paramètre *Streaming Multiprocessors* (SM) active indique le pourcentage de temps pendant lequel le GPU exécute un warp (un groupe de *threads* consécutifs) dans la dernière fenêtre d’échantillonnage. Cette valeur devrait idéalement se situer autour de 80 %. Pour le *SM occupancy* (défini comme le rapport entre le nombre de warps affectés à un SM et le nombre maximal de warps qu’un SM peut gérer), une valeur autour de 50 % est généralement attendue. Concernant le paramètre *Tensor*, la valeur devrait être la plus élevée possible. Idéalement, votre code devrait exploiter cette partie du GPU, optimisée pour les multiplications et convolutions de matrices multidimensionnelles. Enfin, pour les opérations en virgule flottante (*Floating Point*) FP64, FP32 et FP16, vous devriez observer une activité significative sur un seul de ces types, selon la précision utilisée par votre code. 
[thumb|900px|center](File:GPU_cycles_de_calcul_utilisé.png.md)
</div>

<div lang="fr" dir="ltr" class="mw-content-ltr">
À gauche, vous avez un graphique indiquant la mémoire utilisée par le GPU. À droite, un graphique des cycles d'accès du GPU à la mémoire, représentant le pourcentage de cycles pendant lesquels l’interface mémoire de l’appareil est active pour envoyer ou recevoir des données.
[thumb|900px|center](File:Mémoire_GPU.png.md)
</div>

<div lang="fr" dir="ltr" class="mw-content-ltr">
Le graphique de puissance GPU affiche l’évolution de la consommation énergétique (en watts) du GPU au fil du temps. 
[thumb|900px|center](File:Puissance_GPU.png.md)
</div>

<div lang="fr" dir="ltr" class="mw-content-ltr">
À gauche, la bande passante GPU sur le bus PCIe (ou **PCI Express**, pour *Peripheral Component Interconnect Express*). À droite, bande passante GPU sur le bus NVlink. Le bus NVLink est une technologie développée par NVIDIA pour permettre une communication ultra-rapide entre plusieurs GPU.
[thumb|900px|center](File:Bande_passante-GPU.png.md)
</div>

<div lang="fr" dir="ltr" class="mw-content-ltr">
Pour les statistiques des ressources du nœud au complet, sachez quelles peuvent être imprécises si le nœud est partagé entre plusieurs utilisateurs. Le graphique de gauche, illustre l'évolution de la bande passante utilisée par la tâche au fil du temps, en lien avec les logiciels, les licences, etc. Le graphique de droite représente l’évolution de la bande passante réseau utilisée par une tâche ou un ensemble de tâches via le réseau Infiniband, au fil du temps. On peut y observer les périodes de transfert massif de données (ex. : lecture/écriture sur un système de fichiers (Lustre), communication MPI entre nœuds).
[thumb|900px|center](File:Ressources_du_noeud.png.md)
</div>

<div lang="fr" dir="ltr" class="mw-content-ltr">
Le graphique de gauche illustre l’évolution du nombre d’opérations d’entrée/sortie par seconde (IOPS) effectuées sur le disque local au fil du temps. Celui de droite montre l’évolution de la bande passante utilisée sur le disque local au fil du temps, c’est-à-dire la quantité de données lues ou écrites par seconde.
[thumb|900px|center](File:IOPS.png.md)
</div>

Use of local disk space
[thumb|900px|center](File:Espace_utilisé.png.md)

Capacity used
[thumb|900px|center](File:Puissance_utilisé.png.md)

<span id="Statistiques_d&#039;un_compte"></span>
= Account statistics =

<div lang="fr" dir="ltr" class="mw-content-ltr">
La section **Statistique d'un compte** regroupe l'utilisation de votre groupe dans deux sous-sections: CPU et GPU.
[thumb|900px|center](File:Portail_Utilisateur_vos_comptes.png.md)
</div>


<span id="Statistiques_d&#039;un_compte_CPU"></span>
## CPU account statistics

<div lang="fr" dir="ltr" class="mw-content-ltr">
Vous y trouverez la somme des demandes de votre groupe pour les cœurs CPU, ainsi que leur utilisation correspondante au cours des derniers mois. Vous pouvez également suivre l'évolution de votre priorité, qui varie en fonction de votre utilisation.
[thumb|900px|center](File:Utilisation_du_compte.png.md)
</div>

This graph shows the applications that are used more frequently.
[thumb|900px|center](File:Application_used_CPU.png.md)

<div lang="fr" dir="ltr" class="mw-content-ltr">
Vous pouvez consulter ici l'utilisation des ressources par chacun des utilisateurs de votre groupe.  
[thumb|900px|center](File:Utilisation_détaillée_par_utilisateur.png.md)
</div>

This graph shows the CPU cores wasted by each user, over time.
[thumb|900px|center](File:Coeur_CPU_gaspillé.png.md)

<div lang="fr" dir="ltr" class="mw-content-ltr">
Vous pouvez consulter ici l’utilisation de la mémoire par chacun des utilisateurs de votre groupe.
[thumb|900px|center](File:Mémoire_compte.png.md)
</div>

This graph shows the memory wasted by each user.
[thumb|900px|center](File:Mémoire_gaspillée.png.md)

<div lang="fr" dir="ltr" class="mw-content-ltr">
Vous avez ensuite une représentation de votre activité sur les systèmes de fichiers. À gauche, le graphique montre le nombre de commandes d’écriture sur disque que vous avez effectuées. (input/output operations per second (IOPS)) À droite, vous voyez la quantité de données transférées vers les serveurs sur une période donnée. (Bande passante)
[thumb|900px|center](File:Système_de_fichier_compte.png.md)
</div>

<div lang="fr" dir="ltr" class="mw-content-ltr">
Vous avez une liste des dernières tâches qui ont été effectuées pour l'ensemble du groupe.
[thumb|900px|center](File:Tâches_en_cours-1.png.md)
[thumb|900px|center](File:Tâche_en_cours-2.png.md)
</div>

<span id="Statistiques_d&#039;un_compte_GPU"></span>
## GPU account statistics

Here you can see the total GPU requests for your group, along with their usage over the past few months. You can also track your priority, which varies based on your usage. 
[thumb|900px|center](File:Utilisation_compte_GPU_détails.png.md)

This graph shows the software that are more frequently used.
[thumb|900px|center](File:Application_utilisé_compte_GPU.png.md)

Here you see the resources used by each user in your group.
[thumb|900px|center](File:GPU_utilisé_par_utilisateur_compte_GPU.png.md)

This graph shows the quantity of GPUs wasted by each user.
[thumb|900px|center](File:GPU_gaspillé_compte_GPU.png.md)

Here you see the CPU allocated and used by your GPU tasks.
[thumb|900px|center](File:CPU_compte_GPU.png.md)

This graph shows the CPUs wasted by your GPU tasks.
[thumb|900px|center](File:Coeur_CPU_gaspillé_compte_GPU.png.md)

Here you see the memory used by each user in your group.
[thumb|900px|center](File:Mémoire_compte_GPU.png.md)

This graph shows the memory wasted by each user.
[thumb|900px|center](File:Mémoire_gaspillée_GPU.png.md)

<div lang="fr" dir="ltr" class="mw-content-ltr">
Vous avez ensuite une représentation de votre activité sur les systèmes de fichiers. À gauche, le graphique montre le nombre de commandes d’écriture sur disque que vous avez effectuées. (input/output operations per second (IOPS)) À droite, vous voyez la quantité de données transférées vers les serveurs sur une période donnée. (Bande passante)
[thumb|900px|center](File:Système_de_fichier_GPU.png.md)
</div>

Here you see the last tasks that were run by your group.
[thumb|900px|center](File:Tâches_en_cours-1.png.md)
[thumb|900px|center](File:Tâche_en_cours-2.png.md)

<span id="Statistiques_du_cloud"></span>
= Cloud statistics =

<div lang="fr" dir="ltr" class="mw-content-ltr">
Le premier tableau « Vos instances » présente l'ensemble des machines virtuelles associées à un compte. La colonne « Saveur » fait référence au [type de machine virtuelle](Virtual_machine_flavors/fr.md). La colonne « UUID » correspond à un identifiant unique attribué à chaque machine virtuelle.
</div>

[thumb|900px|center](File:Tableau_vos_instances.png.md)

Then, each virtual machine has its own usage statistics (CPU cores, memory, disk bandwidth, IOPS and network bandwidth) that can be shown for the last month, week, day or hour.

[thumb|900px|center](File:Coeurs_CPU.png.md)

[thumb|900px|center](File:Mémoire_cloud.png.md)

[thumb|900px|center](File:Bande_passante_disque_cloud.png.md)

[thumb|900px|center](File:IOPS_disque.png.md)

[thumb|900px|center](File:Bande_passante_réseau_cloud.png.md)