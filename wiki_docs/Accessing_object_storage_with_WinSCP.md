# Accessing object storage with WinSCP

This page contains instructions on how to set up and access [Arbutus object storage](Arbutus_object_storage.md) with WinSCP, one of the [object storage clients](Arbutus_object_storage_clients.md) available for this storage type.

## Installing WinSCP 
WinSCP can be installed from https://winscp.net/.

## Configuring WinSCP
Under "New Session", make the following configurations:
<ul>
<li>File protocol: Amazon S3</li>
<li>Host name: object-arbutus.cloud.computecanada.ca</li>
<li>Port number: 443</li>
<li>Access key ID: 20_DIGIT_ACCESS_KEY</li>
</ul>
and "Save" these settings as shown below

[600px|thumb|center|WinSCP configuration screen](File:WinSCP_Configuration.png.md)

Next, click on the "Edit" button and then click on "Advanced..." and navigate to "Environment" to "S3" to "Protocol options" to "URL style:" which <b>must</b> changed from "Virtual Host" to "Path" as shown below:

[600px|thumb|center|WinSCP Path Configuration](File:WinSCP_Path_Configuration.png.md)

This "Path" setting is important, otherwise WinSCP will not work and you will see hostname resolution errors, like this:
[400px|thumb|center|WinSCP resolve error](File:WinSCP_resolve_error.png.md)

## Using WinSCP
Click on the "Login" button and use the WinSCP GUI to create buckets and to transfer files:

[800px|thumb|center|WinSCP file transfer screen](File:WinSCP_transfers.png.md)

## Access Control Lists (ACLs) and Policies
Right-clicking on a file will allow you to set a file's ACL, like this:
[400px|thumb|center|WinSCP ACL screen](File:WinSCP_ACL.png.md)