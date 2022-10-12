# COVID-CH-dashboard
Streamlit dashboard with analysis of cases and hospitalizations from Genebra. 

## Running it locally

```bash
$ streamlit run dashboard.py --server.port 8502
```

### Run with docker-compose :

**Pre-requisites**
* Docker installed and running
* docker-compose installed

It also works fine with podman and [podman-compose].(https://phoenixnap.com/kb/podman-compose#ftoc-heading-3)

```bash
$ docker-compose -f docker/docker-compose.yml up
# or 
$ podman-compose -f docker/docker-compose.yml up

# When dependencies change and you need to force a rebuild
$ docker-compose -f docker/docker-compose.yml up --build

# When finished
$ docker-compose down
```

You can see the app running at http://localhost:8501
