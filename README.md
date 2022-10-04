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

```bash
$ docker-compose up

# When dependencies change and you need to force a rebuild
$ docker-compose up --build

# When finished
$ docker-compose down
```
