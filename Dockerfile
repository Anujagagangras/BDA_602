FROM ubuntu

RUN mkdir /results

COPY ./Final_Project.sh /Final_Project.sh
COPY ./baseball.sql /baseball.sql
COPY ./Final_Project.sql /Final_Project.sql
COPY ./Final_Project.py /Final_Project.py
COPY ./requirements.txt /requirements.txt

RUN chown 1000:1000 ./

RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
    wget \
    curl \
    ca-certificates

RUN wget https://downloads.mariadb.com/MariaDB/mariadb_repo_setup
RUN echo "367a80b01083c34899958cdd62525104a3de6069161d309039e84048d89ee98b  mariadb_repo_setup" \
    | sha256sum -c -
RUN chmod +x mariadb_repo_setup
RUN ./mariadb_repo_setup --mariadb-server-version="mariadb-10.6" --skip-check-installed

RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     python3 python3-dev \
     build-essential libssl-dev libffi-dev \
     libxml2-dev libxslt1-dev zlib1g-dev \
     python3-pip \
     libmariadb3 \
     libmariadb-dev \
     mariadb-client 

RUN apt upgrade --no-install-recommends --yes
RUN rm -rf /var/lib/apt/lists/*
RUN pip install Jinja2 

RUN pip3 install --compile --no-cache-dir -r ./requirements.txt

RUN chmod +x ./Final_Project.sh
CMD ./Final_Project.sh
