# TEST_DS_GB
Тестовое задание на вакансию Data Scientist в GlowByte, Новиков Георгий https://hh.ru/resume/0de115c7ff06aa4f360039ed1f62633362424e

Основная идея решения заключается в том, что мы хотим предсказать, на сколько был выработан ресурс мотора к данному циклу. Для этого в качестве целевой переменной возьмём (номер текущего цикла) / (номер последнего цикла) для данного мотора. Для этого обучим несколько моделей регрессии (estimate_loss.py) и придём к выводу, что лучшей является метод опорных векторов по оценке MSE. На train/test делим 9/1, но для более корректного обучения делим не по строкам обучающей выборки, а по моторам. Дальнейшие решения зависят от нескольких факторов:
1) насколько больше времени (и материальных ресурсов) займёт техобслуживание мотора и его возвращение в эксплуатацию по сравнению с полным отказом мотора, его ремонтом и возвращением в эксплуатацию или покупкой нового
2) насколько линейным можно считать ресурс мотора, так как изначально при задании целевой переменной мы считаем, что ресурс мотора линеен
3) возможен ли в принципе ремонт мотора, или он целиком является скорее расходным материалом на производстве

Если ремонт мотора невозможен, то задача сводится к тому, при каком пороговом значении стоит заказать новый мотор, чтобы он пришёл точно в срок поломки старого мотора. Для определения данного значения необходимо знать, сколько времени будет идти доставка нового мотора, но получаемая выгода от такого предсказания довольно мала, так как проще заказать моторы заранее.
Если ремонт мотора возможен, то вопрос заключается в том, при каком пороговом значении предсказывающей модели снимать мотор с производственной линии, и это зависит в том числе от линейности ресурса мотора и времени, необходимого на его починку, а также от наличия запасных моторов в наличии, которые можно поставить на линию на время техобслуживания. Если запасные моторы имеются, то нет смысла сильно рисковать, и лучше выбрать пороговое значение не больше 0.8 для отправки мотора на техобслуживание. Если же запаса нет, то вопрос исключительно в разнице времени между полноценным ремонтом и техобслуживанием: чем разница больше, тем меньше стоит выбрать пороговое значение.
Файл predict_lib подключается на ПК, имеющий доступ к текущим данным моторов, и импортируется в существующий там код, чтобы подгрузить обученную заранее модель (fitted_model.sav) и использовать её для получения этого самого значения ресурса двигателя. Можно было бы подумать над реализацией дообучения, но если наша модель будет достаточно хороша, то моторы не будут ломаться, и корректных данных для дообучения мы больше не получим, ведь настоящий ресурс мотора, отправленного на техобслуживание, мы не узнаем. А если данные перестанут подходить модели (что неизбежно), можно будет в рамках поддержки обучить на более свежих данных новую модель.

Доп. задание: можно было бы попробовать предсказывать наиболее оптимальные настройки, которые наиболее благоприятно влияют на ресурс двигателя, чтобы их выставлять, но такая постановка задачи актуальна только в случае, если эти настройки являются не функциональными, а эксплуатационными.
