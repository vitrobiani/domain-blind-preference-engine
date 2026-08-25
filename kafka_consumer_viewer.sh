#! /bin/sh


docker exec -it preference-engine-kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic preference-engine.interactions --property print.timestamp=true
