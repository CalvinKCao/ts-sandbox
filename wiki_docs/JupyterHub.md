# JupyterHub

<i>JupyterHub is the best way to serve Jupyter Notebook for multiple users. It can be used in a class of students, a corporate data science group or scientific research group.</i>
<ref>http://jupyterhub.readthedocs.io/en/latest/index.html</ref>

JupyterHub provides a preconfigured version of JupyterLab and/or Jupyter Notebook; for more configuration options, please check the [Jupyter](Jupyter.md) page.

{{Warning
|title=Running notebooks
|content=Jupyter Lab and notebooks are meant for **short** interactive tasks such as testing, debugging or quickly visualize data (few minutes). Running longer analysis must be done in an [non-interactive job (sbatch)](Running_jobs#Use_sbatch_to_submit_jobs.md).
See also [how to run notebooks as python scripts below](JupyterHub#Running_notebooks_as_Python_scripts.md).
}}
= Alliance initiatives =

Some regional initiatives offer access to computing resources through JupyterHub.

## JupyterHub on clusters

On the following clusters[<sup>‡</sup>](#clusters_note.md), use your Alliance username and password to connect to JupyterHub:
{| class="wikitable"
|-
! JupyterHub !! Comments
|-
| <b>[Fir](https://jupyterhub.fir.alliancecan.ca/)</b> || Provides access to JupyterLab servers spawned through jobs on the [Fir](Fir.md) cluster. 
|-
| <b>[Narval](https://jupyterhub.narval.alliancecan.ca/)</b> || Provides access to JupyterLab servers spawned through jobs on the [Narval](Narval.md) cluster.
|-
| <b>[Rorqual](https://jupyterhub.rorqual.alliancecan.ca/)</b> || Provides access to JupyterLab servers spawned through jobs on the [Rorqual](Rorqual.md) cluster.
|}

Some clusters provide access to JupyterLab through Open OnDemand. See [JupyterLab](JupyterLab.md) for more information. 

<b><sup id="clusters_note">‡</sup> Note that the compute nodes running the Jupyter kernels do not have internet access</b>. This means that you can only transfer files from/to your own computer; you cannot download code or data from the internet (e.g. cannot do "git clone", cannot do "pip install" if the wheel is absent from our [wheelhouse](Available_Python_wheels.md)). You may also have problems if your code performs downloads or uploads (e.g. in machine learning where downloading data from the code is often done).

## JupyterHub for universities and schools

- The [Pacific Institute for the Mathematical Sciences](https://www.pims.math.ca) in collaboration with the Alliance and [Cybera](http://www.cybera.ca) offer cloud-based hubs to universities and schools. Each institution can have its own hub where users authenticate with their credentials from that institution. The hubs are hosted on Alliance [clouds](Cloud.md) and are essentially for training purposes. Institutions interested in obtaining their own hub can visit [Syzygy](http://syzygy.ca).

= Server options =

[thumb|<i>Server Options</i> form on Béluga's JupyterHub](File:JupyterHub_Server_Options.png.md)
Once logged in, depending on the configuration of JupyterHub, the user's web browser is redirected to either
<b>a)</b> a previously launched Jupyter server,
<b>b)</b> a new Jupyter server with default options, or
<b>c)</b> a form that allows a user to set different options for their Jupyter server before pressing the <i>Start</i> button.
In all cases, it is equivalent to accessing requested resources via an [interactive job](Running_jobs#Interactive_jobs.md) on the corresponding cluster.

<b>Important:</b> On each cluster, only one interactive job at a time gets a priority increase in order to start in a few seconds or minutes. That includes <code>salloc</code>, <code>srun</code> and JupyterHub jobs. If you already have another interactive job running on the cluster hosting JupyterHub, your new Jupyter session may never start before the time limit of 5 minutes.

## Compute resources

For example, <i>Server Options</i> available on [Béluga's JupyterHub](https://jupyterhub.beluga.computecanada.ca/) are:
- <i>Account</i> to be used: any <code>def-*</code>, <code>rrg-*</code>, <code>rpp-*</code> or <code>ctb-*</code> account a user has access to
- <i>Time (hours)</i> required for the session
- <i>Number of (CPU) cores</i> that will be reserved on a single node
- <i>Memory (MB)</i> limit for the entire session
- (Optional) <i>GPU configuration</i>: at least one GPU
- <i>[User interface](JupyterHub#User_Interface.md)</i> (see below)

## User interface

While JupyterHub allows each user to use one Jupyter server at a time on each hub, there can be multiple options under <i>User interface</i>:
- <b>[JupyterLab](JupyterLab.md)</b> (modern interface): This is the most recommended Jupyter user interface for interactive prototyping and data visualization.
- Jupyter Notebook (classic interface): Even though it offers many functionalities, the community is moving towards [JupyterLab](JupyterHub#JupyterLab.md), which is a better platform that offers many more features.
- Terminal (for a single terminal only): It gives access to a terminal connected to a remote account, which is comparable to connecting to a server through an SSH connection.

Note: JupyterHub could also have been configured to force a specific user interface. This is usually done for special events.

= JupyterLab =

The JupyterLab interface is now described in our [JupyterLab](JupyterLab.md) page.

= Possible error messages =

## Spawn failed: Timeout

Most JupyterHub errors are caused by the underlying job scheduler which is either unresponsive or not able to find appropriate resources for your session. For example:

[thumb|upright=1.1|JupyterHub - Spawn failed: Timeout](File:JupyterHub_Spawn_failed_Timeout.png.md)
- When starting a new session, JupyterHub automatically submits on your behalf a new [interactive job](Running_jobs#Interactive_jobs.md) to the cluster. If the job does not start within five minutes, a "Timeout" error message is raised and the session is cancelled.
** Just like any interactive job on any cluster, a longer requested time can cause a longer wait time in the queue. Requesting a GPU or too many CPU cores can also cause a longer wait time. Make sure to request only the resources you need for your session.
** If you already have another interactive job on the same cluster, your Jupyter session will be waiting along with other regular batch jobs in the queue. If possible, stop or cancel any other interactive job before using JupyterHub.
** There may be just no resource available at the moment. Check the [status page](https://status.alliancecan.ca/) for any issue and try again later.

## Authentication error: Error 403

Your account or your access to the cluster is currently inactive:
1. Make sure your account is active, that is [<b>it has been renewed</b>](https://alliancecan.ca/en/services/advanced-research-computing/account-management/account-renewals)
1. Make sure your [<b>access to a cluster</b>](https://ccdb.alliancecan.ca/me/access_services) is enabled

= References =