FROM ubuntu

RUN mkdir /scripts
RUN mkdir /results

COPY ./Homework_6.sh /scripts/Homework_6.sh
COPY ./baseball.sql /scripts/baseball.sql
COPY ./Homework_6.sql /scripts/Homework_6.sql
RUN chown 1000:1000 /scripts

RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     mysql-client \
     wget \
  && rm -rf /var/lib/apt/lists/*

RUN chmod +x /scripts/Homework_6.sh
CMD ./scripts/Homework_6.sh
