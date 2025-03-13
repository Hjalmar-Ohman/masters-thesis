## Connect your VM to local VSCode

Get the output of running command

```sh
Amin: gcloud compute ssh <notebook-instance> --dry-run
Rahul: gcloud compute ssh --zone "europe-west1-b" "hjalmar-masters-thesis" --tunnel-through-iap --project "chart-rag-5cab6dd3" --dry-run
```

You need the output to correctly set your `config` file in `.ssh` folder in the home directory.

```sh
Host <notebook-instance-name> # Just a name does not matter
  HostName compute.<numbers> # Get that from the end of --dry-run command
  IdentityFile C:\Users\<s-id>\.ssh\google_compute_engine
  UserKnownHostsFile C:\Users\<s-id>\.ssh\google_compute_known_hosts
  User <firstname_lastname_seb_se> #your email, change all . and @ to underscore
  ProxyCommand "C:\\Users\\<s-id>\\<path-to-google-cloud-sdk>\\bundledpython\\python.exe" "-S" "C:\\Users\\<s-id>\\<path-to-google-cloud-sdk>\\lib\\gcloud.py" compute start-iap-tunnel "<notebook-instance-name>" "%p" --listen-on-stdin --project=<project-id> --zone=<vm-zone> --verbosity=warning
  RemoteCommand sudo su jupyter
```

## Install VSCode
After downloading VSCode from [here](https://code.visualstudio.com/download) and local installation. 
- Open extensions and search for `Remote-SSH` and `Remote Explorer` and install them.
- You can see the name of your instance in the list on the block on the right panel under `Remote Explorer` click on the vm to connect.
- Start coding and exploration in your remote instance.

## Problem of `Jupyter` user
In general the last line of the config file `RemoteCommand sudo su jupyter` must enable us to connect to the vm as `jupyter` user. But to do that you must enable `Enable Remote Command` in your VSCode setting. However on SEB machine this recipe has not been successful.


## Status on GPU
nvidia-smi