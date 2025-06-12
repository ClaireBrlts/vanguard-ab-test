WITH

demographics AS (SELECT  clients.client_id, clients.Variation AS variation, clnt_age AS age, gendr AS gender, gen.generation
FROM clients
JOIN (SELECT client_id, Variation, generation FROM client_generation) AS gen
ON clients.client_id = gen.client_id),

errs AS (SELECT visit_errors.client_id, visit_errors.visit_id, visit_errors.process_step, visit_errors.step_validation
FROM visit_errors)

SELECT d.client_id, d.variation, d.age, d.gender, d.generation, e.visit_id, e.process_step, e.step_validation
FROM demographics AS d
JOIN errs AS e
ON d.client_id = e.client_id;

WITH

demographics AS (SELECT  clients.client_id, clients.Variation AS variation, clnt_age AS age, gendr AS gender, gen.generation
FROM clients
JOIN (SELECT client_id, Variation, generation FROM client_generation) AS gen
ON clients.client_id = gen.client_id),

duration AS (SELECT client_id, visit_id, process_step, date_time, duration
FROM visit_duration)

SELECT de.client_id, de.variation, de.age, de.gender, de.generation, du.visit_id, du.date_time, du.duration
FROM duration AS du
JOIN demographics AS de
ON de.client_id = du.client_id;


