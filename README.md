# mle-template-case-sprint2

Добро пожаловать в репозиторий-шаблон Практикума для проекта 2 спринта. Ваша цель — улучшить ключевые метрики модели для предсказания стоимости квартир Яндекс Недвижимости.

Полное описание проекта хранится в уроке «Проект. Улучшение baseline-модели» на учебной платформе.

Здесь укажите имя вашего бакета: 's3-student-mle-20241021-ea9d7e02d5'

#*********************************************************************
Этап 1: Разворачивание MLflow и регистрация модели

процесс развертывания MLFLOW - в терминале
cd ~/mle_projects/mle-project-sprint-2
# обновление локального индекса пакетов
sudo apt-get update
# установка расширения для виртуального пространства
sudo apt-get install python3.10-venv
# создание виртуального пространства
python3.10 -m venv .venv_sprint2

source .venv_sprint2/bin/activate
# файл с нужными библиотеками уже подготовлен заранее  
pip install -r requirements.txt

# для работы jupiter notebook делаю kernel
python -m ipykernel install --user --name=.venv_sprint2

# далее запускаю shell файл
cd ~/mle_projects/mle-project-sprint-2/mlflow_server/
sh run_mlflow.sh

# mlflow доступен по ссылке
http://127.0.0.1:5000

# если надо перезагрузить
pkill gunicorn

# Регистрация модели


#*********************************************************************
Этап 2: Проведение EDA
результаты EDA доступны по ссылке mle_projects/mle-project-sprint-2/model_improvement/project_template_sprint_2.ipynb

основные выводы по таблице с квартирами
 - в таблице есть бинарные (is_apartment,studio) и числовые (floor,kitchen_area,living_area,rooms,total_area) признаки
 - бинарные признаки studio - не заполнен, is_apartment - заполнен
 - этаж floor - от 1 до 56, нет пропусков
 - площадь кухни и жилых помещений kitchen_area, living_area - есть пропуски, надо восстанавливать значения
 - количество комнат rooms - от 1 до 20, есть выбросы (редкие большие значения)
 - общая площадь total_area - от 11 до 960 метров, есть выбросы (редкие большие значения)
 - цена (целевой признак) - есть выбросы
 - явных дублей нет
 - есть неявные дубли (необходимо чистить)

 основные выводы по таблице с домами
 - в таблице есть бинарные (has_elevator) , категориальные (building_type_int) и числовые (build_year, latitude, longitude, ceiling_height, flats_count, floors_total) признаки
 - бинарный признак has_elevator - заполнен хорошо
 - build_year - c 1900 года
 - гео признаки - latitude и longitude - напрямую использовать в модели не получится, можно выделить районы и посмотреть зависимость цены от района
 - ceiling_height, flats_count, floors_total - есть выбросы в данных
 - неявных и явных дублей в таблице с домами нет

объединение в 1 датасет
выводы по датасету
1 - есть пропуски и выбросы в данных, есть дубли
2 - есть признаки, которые нужно обрабатывать (география, тип здания)
3 - из предварительного анализа самые важные признаки, которые влияют на цену
 - цена положительно зависит от площади, от количества комнат, от размера потолка и слабо отрицательно от количества квартир в доме
 - наличие лифта, этажность, признак апартаментов почти не влияют на цену

Основные выводы по результатам EDA по улучшению модели
1 - заполнить пропуски по living area, kitchen area и посмотреть насколько важными являются признаки
2 - создать признак район, для модели выделить район, который отличается ценами
3 - использовать тип здания в модели
4 - добавить признак 1 и последний этаж (по результатам EDA квартиры на 1 этаже дешевле)

#*********************************************************************
Этап 3: Генерация признаков и обучение модели
1 - заполнил пропуски по living area и kitchen area
2 - сделал преобразование OneHotEncoder для 'building_type_int'
3 - рассчитал признак 1й и последний этаж 

метрики улучшились
модель0 - базовая  - {'r2': 0.6658816312057878, 'mse': 49198764900740.38}
модель1 - новые признаки - {'r2': 0.7187323105446903, 'mse': 40475464284352.66}

в модели получилось 20+ признаков, на следующем этапе проверяю все ли важны

#*********************************************************************
Этап 4: Отбор признаков и обучение новой версии модели

метод 1 - sequential forward feature selection (метрика 'neg_mean_squared_error')
метод 2 - sequential backward feature selection (метрика 'neg_mean_squared_error')

далее объединяю 2 списка признаков и смотрю на модель
модель 2 - отобранные признаки - {'r2': 0.7105984859677652, 'mse': 41181795480470.0}

качество почти не изменилось когда оставили половину признаков
такую модель значительно проще интерпретировать
Самые важные признаки - это район и общая площадь, старые дома ценятся выше новых

#*********************************************************************
Этап 5: Подбор гиперпараметров и обучение новой версии модели

подбираю гиперпараметр "alpha" при помощи optuna
метод 1 - оптимизация метрики r2
метод 2 - оптимизация метрики 'mse'

модель 3 - optuna r2 {'r2': 0.710595634218038, 'mse': 41182201284062.36}

вывод - гиперпараметр на качество модели влияет незначительно



