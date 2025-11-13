import logging
import os
import re
import time
from datetime import datetime
from sqlite3 import connect
from typing import Dict, List, Optional, Tuple, Union

import gigaam
import noisereduce as nr
import numpy as np
import sounddevice as sd
import torch
from matplotlib.patches import Patch
from matplotlib import pyplot as plt
from scipy.io.wavfile import write
from tabulate import tabulate

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Конфигурация
SAMPLE_RATE_ASR = 16_000
SAMPLE_RATE_TTS = 48_000
SPEAKER = "baya"
RECORD_DURATION = 10
NOISE_REDUCTION = True
DB_PATH = "orders.db"
AUDIO_LOGS_DIR = "audio_logs"
BRAKED_ZONE = "ряд пятнадцать стеллаж два"
PICKING_DELAY = 10

# Размеры склада
NUM_ROWS = 15
NUM_RACKS_PER_ROW = 2
NUM_TIERS = 4
NUM_CELLS = 15

# Создание директории для аудиозаписей
os.makedirs(AUDIO_LOGS_DIR, exist_ok=True)

# Определение устройства
device = "cuda" if torch.cuda.is_available() else "cpu"

# Инициализация моделей
try:
    logger.info("Инициализация моделей...")
    tts_model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language="ru",
        speaker="v4_ru",
    )
    tts_model.to(device)
    asr_model = gigaam.load_model("v2_ctc").to(device)
except Exception as e:
    logger.error(f"Ошибка инициализации моделей: {e} (!)")
    exit(1)

# Цвета для статусов
STATUS_COLORS = {
    "pending": "white",
    "completed": "green",
    "defective": "red",
    "not_found": "yellow",
    "partial": "orange",
    "error": "gray",
}


def digits_to_words(number_str: str) -> str:
    number_map = {
        "0": "ноль", "1": "один", "2": "два", "3": "три", "4": "четыре",
        "5": "пять", "6": "шесть", "7": "семь", "8": "восемь", "9": "девять",
        "10": "десять", "11": "одиннадцать", "12": "двенадцать",
        "13": "тринадцать", "14": "четырнадцать", "15": "пятнадцать",
    }
    if number_str in number_map:
        return number_map[number_str]
    return " ".join(number_map[digit] for digit in number_str if digit in number_map)


def normalize_number(text: str) -> str:
    number_map = {
        "ноль": "0", "один": "1", "одна": "1", "два": "2", "две": "2",
        "три": "3", "четыре": "4", "пять": "5", "шесть": "6", "семь": "7",
        "восемь": "8", "девять": "9", "десять": "10", "одиннадцать": "11",
        "двенадцать": "12", "тринадцать": "13", "четырнадцать": "14", "пятнадцать": "15",
    }
    words = re.split(r"\s+", text.strip())
    normalized_words = [number_map.get(word, word) for word in words]
    return " ".join(normalized_words)


def save_audio(audio_data: np.ndarray, sample_rate: int, prefix: str, order_id: str, position_id: str = None) -> str:
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{order_id}"
        if position_id:
            filename += f"_{position_id}"
        filename += f"_{timestamp}.wav"
        filepath = os.path.join(AUDIO_LOGS_DIR, filename)
        write(filepath, sample_rate, audio_data)
        logger.info(f"Аудиозапись сохранена: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Ошибка сохранения аудиозаписи: {e}")
        return ""


def plot_warehouse(
    order_data: List[Tuple],
    current_position: Optional[Tuple] = None,
    row_filter: int = 1,
    highlight_mode: str = "status"
) -> None:
    # Сброс Matplotlib
    plt.clf()
    plt.close('all')
    plt.rcdefaults()  # Сброс настроек Matplotlib
    fig, ax = plt.subplots(figsize=(12, 6))

    # Цвета в RGB (значения от 0 до 1)
    color_map = {
        'white': (1.0, 1.0, 1.0),    # pending
        'green': (0.0, 1.0, 0.0),    # completed
        'red': (1.0, 0.0, 0.0),      # defective
        'yellow': (1.0, 1.0, 0.0),   # not_found
        'orange': (1.0, 0.647, 0.0), # partial
        'gray': (0.5, 0.5, 0.5)      # error
    }

    # Создаём матрицу RGB-цветов для каждой ячейки
    grid = np.ones((NUM_TIERS, NUM_RACKS_PER_ROW * NUM_CELLS, 3))  # Инициализируем белым (1, 1, 1)
    positions_found = False

    # Заполнение grid
    for order in order_data:
        _, pos_id, row, rack, tier, cell, _, _, status = order
        row, rack, tier, cell = int(row), int(rack), int(tier), int(cell)
        if row == row_filter:
            positions_found = True
            status = str(status).strip().lower()
            logger.info(f"Обработка позиции {pos_id}: статус={status}")
            # Определение статуса
            if status.startswith('partial_'):
                status_key = 'partial'
            elif status in ['pending', 'completed', 'defective', 'not_found', 'error']:
                status_key = status
            else:
                status_key = 'pending'
                logger.warning(f"Неизвестный статус '{status}' для позиции {pos_id}, установлен pending")
            color = STATUS_COLORS.get(status_key, "white")
            x_idx = (rack - 1) * NUM_CELLS + (cell - 1)
            y_idx = tier - 1
            grid[y_idx, x_idx] = color_map[color]
            logger.info(f"Позиция {pos_id}: status_key={status_key}, color={color}")

    if not positions_found:
        logger.warning(f"Нет позиций для ряда {row_filter}")

    # Отрисовка
    ax.imshow(grid, origin="lower", extent=(-0.5, NUM_RACKS_PER_ROW * NUM_CELLS - 0.5, -0.5, NUM_TIERS - 0.5))

    # Оси
    ax.set_xticks(np.arange(NUM_RACKS_PER_ROW * NUM_CELLS))
    ax.set_yticks(np.arange(NUM_TIERS))
    xticklabels = [f"{r}.{c}" for r in range(1, NUM_RACKS_PER_ROW + 1) for c in range(1, NUM_CELLS + 1)]
    yticklabels = [f"Ярус {i+1}" for i in range(NUM_TIERS)]
    ax.set_xticklabels(xticklabels, rotation=90)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel("Стеллаж.Ячейка")
    ax.set_ylabel("Ярусы")
    ax.set_title(f"Схема склада (Ряд {row_filter})")

    # Сетка
    ax.set_xticks(np.arange(-0.5, NUM_RACKS_PER_ROW * NUM_CELLS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, NUM_TIERS, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)

    # Синяя рамка только в режиме highlight_mode="init"
    if current_position and highlight_mode == "init":
        _, pos_id, row, rack, tier, cell, _, _, _ = current_position
        row, rack, tier, cell = int(row), int(rack), int(tier), int(cell)
        if row == row_filter:
            x_idx = (rack - 1) * NUM_CELLS + (cell - 1)
            y_idx = tier - 1
            ax.add_patch(plt.Rectangle(
                (x_idx - 0.5, y_idx - 0.5), 1, 1,
                fill=False, edgecolor="blue", linewidth=2
            ))

    # Легенда
    legend_elements = [Patch(facecolor=color_map[color], label=status) for status, color in STATUS_COLORS.items()]
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.15, 1))

    # Отступы
    plt.margins(0)
    plt.tight_layout()

    # Сохранение для отладки
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(AUDIO_LOGS_DIR, f"warehouse_row{row_filter}_{timestamp}.png")
    plt.savefig(filepath, bbox_inches="tight", dpi=150)
    logger.info(f"График сохранён: {filepath}")

    # Отображение в окне
    plt.show(block=False)
    plt.pause(1.0)
    plt.close()


def print_order_table(order_data: List[Tuple]) -> None:
    headers = ["Заказ", "Позиция", "Ряд", "Стеллаж", "Ярус", "Ячейка", "Товар", "Кол-во", "Статус"]
    table_data = [
        [order_id, pos_id, row, rack, tier, cell, item, qty, status]
        for order_id, pos_id, row, rack, tier, cell, item, qty, status in order_data
    ]
    logger.info("Вывод таблицы заказов")
    print("\nТекущее состояние заказа:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


class OrderDatabase:
    def __init__(self, db_path: str):
        self.conn = connect(db_path)
        self.cursor = self.conn.cursor()
        self.init_db()

    def init_db(self) -> None:
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT,
                position TEXT,
                row TEXT,
                rack TEXT,
                tier TEXT,
                cell TEXT,
                item TEXT,
                quantity TEXT,
                status TEXT DEFAULT 'pending'
            )
            """
        )
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS audio_logs (
                order_id TEXT,
                position_id TEXT,
                command_audio_path TEXT,
                response_audio_path TEXT,
                command_text TEXT,
                response_text TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.commit()

    def get_order(self, order_id: Union[str, int]) -> List[Tuple]:
        self.cursor.execute(
            "SELECT * FROM orders WHERE order_id = ? AND status = 'pending'",
            (order_id,),
        )
        return self.cursor.fetchall()

    def update_status(self, order_id: Union[str, int], position: Union[str, int], status: str) -> None:
        self.cursor.execute(
            "UPDATE orders SET status = ? WHERE order_id = ? AND position = ?",
            (status, order_id, position),
        )
        self.conn.commit()

    def save_audio_log(
            self,
            order_id: Union[str, int],
            position_id: Union[str, int],
            command_audio_path: str,
            response_audio_path: str,
            command_text: str,
            response_text: str,
    ) -> None:
        self.cursor.execute(
            """
            INSERT INTO audio_logs (
                order_id, position_id, command_audio_path, response_audio_path,
                command_text, response_text
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (order_id, position_id, command_audio_path, response_audio_path, command_text, response_text),
        )
        self.conn.commit()

    def get_all_orders(self, order_id: Union[str, int]) -> List[Tuple]:
        self.cursor.execute(
            "SELECT order_id, position, row, rack, tier, cell, item, quantity, status FROM orders WHERE order_id = ?",
            (order_id,)
        )
        return self.cursor.fetchall()

    def close(self) -> None:
        self.conn.close()

class VoiceOrderSystem:
    def __init__(self, db_path: str):
        self.db = OrderDatabase(db_path)
        self.response_types = [
            "Готово",
            "Товар не найден",
            "Не хватает единиц товара",
            "Повтори команду",
            "Товар бракован",
            "Принято",
        ]
        self.verification = "Готово"

    def generate_command_text(
            self, row: Union[str, int], rack: Union[str, int], tier: Union[str, int],
            cell: Union[str, int], item: Union[str, int], quantity: Union[str, int]
    ) -> str:
        row_text = digits_to_words(str(row))
        rack_text = digits_to_words(str(rack))
        tier_text = digits_to_words(str(tier))
        cell_text = digits_to_words(str(cell))
        item_text = digits_to_words(str(item))
        quantity_text = digits_to_words(str(quantity))
        return (
            f"Ряд {row_text}, стеллаж {rack_text}, ярус {tier_text}, ячейка {cell_text}, "
            f"товар {item_text}, {quantity_text} единицы"
        )

    def synthesize_speech(self, text: str, order_id: str = None, position_id: str = None) -> Tuple[bool, str]:
        try:
            logger.info(f"Синтез команды: {text}...")
            start_time = time.time()
            audio = tts_model.apply_tts(
                text=text,
                speaker=SPEAKER,
                sample_rate=SAMPLE_RATE_TTS,
            )
            audio_data = np.array(audio, dtype=np.float32)
            audio_path = save_audio(audio_data, SAMPLE_RATE_TTS, "command", order_id, position_id)
            sd.play(audio_data, samplerate=SAMPLE_RATE_TTS)
            sd.wait()
            logger.info(f"Время синтеза речи: {time.time() - start_time:.2f} сек.")
            return True, audio_path
        except Exception as e:
            logger.error(f"Ошибка синтеза речи: {e} (!)")
            return False, ""

    def record_audio(self, order_id: str, position_id: str = None) -> Optional[Tuple[str, np.ndarray]]:
        try:
            logger.info("Запись ответа...")
            recording = sd.rec(
                int(RECORD_DURATION * SAMPLE_RATE_ASR),
                samplerate=SAMPLE_RATE_ASR,
                channels=1,
            )
            sd.wait()
            audio = np.squeeze(recording)
            if NOISE_REDUCTION:
                audio = nr.reduce_noise(
                    y=audio,
                    sr=SAMPLE_RATE_ASR,
                    stationary=True,
                    prop_decrease=0.75,
                )
            audio_path = save_audio(audio, SAMPLE_RATE_ASR, "response", order_id, position_id)
            return audio_path, audio
        except Exception as e:
            logger.error(f"Ошибка записи аудио: {e} (!)")
            return None

    def recognize_speech(self, audio_file: str) -> str:
        try:
            logger.info("Распознавание речи...")
            start_time = time.time()
            transcription = asr_model.transcribe(audio_file)
            logger.info(f"Транскрипция: {transcription}.")
            logger.info(f"Время распознавания: {time.time() - start_time:.2f} сек.")
            return transcription
        except Exception as e:
            logger.error(f"Ошибка распознавания речи: {e} (!)")
            return ""

    def extract_response_data(self, response: str) -> Dict[str, Union[str, Dict]]:
        response = response.strip()
        extracted: Dict[str, Union[str, Dict]] = {"response_type": None, "entities": {}}
        for resp_type in self.response_types:
            if resp_type.lower() in response.lower():
                extracted["response_type"] = resp_type
                break
        if not extracted["response_type"]:
            extracted["response_type"] = "Неизвестный тип ответа"
        if extracted["response_type"] == "Готово":
            response = normalize_number(response)
            numbers = re.findall(r"\d+", response)
            if len(numbers) >= 6:
                item_code = numbers[4] if len(numbers[4]) >= 4 else "".join(numbers[4: len(numbers) - 1])
                extracted["entities"] = {
                    "row": numbers[0],
                    "rack": numbers[1],
                    "tier": numbers[2],
                    "cell": numbers[3],
                    "item": item_code,
                    "quantity": numbers[-1],
                }
                logger.info(f"Извлечено: {extracted['entities']}.")
            elif len(numbers) > 6:
                logger.warning(f"Слишком много чисел в ответе: названы числа {numbers} (!)")
                self.synthesize_speech("Избыточная информация. Пожалуйста, повторите")
            else:
                logger.warning(f"Недостаточно чисел в ответе: названы числа {numbers} (!)")
                self.synthesize_speech("Неполная информация. Пожалуйста, повторите")
        else:
            logger.info(f"Тип ответа '{extracted['response_type']}': сущности не извлекаются.")

        return extracted

    def validate_response(self, response_entities: Dict, expected: Dict) -> bool:
        keys_to_validate = ["row", "rack", "tier", "cell", "item", "quantity"]
        for key in keys_to_validate:
            response_value = response_entities.get(key, "")
            expected_value = str(expected.get(key, ""))
            if not response_value:
                logger.warning(f"Отсутствует значение для {key}: ожидалось {expected_value} (!)")
                return False
            if str(response_value).strip() != expected_value.strip():
                logger.warning(
                    f"Несоответствие в {key}: ожидалось {expected_value}, "
                    f"получено {response_value} (!)"
                )
                return False
        return True

    def process_order(self, order_id: Union[str, int]) -> None:
        order = self.db.get_order(order_id)
        if not order:
            logger.error(f"Заказ {order_id} не найден (!)")
            self.synthesize_speech("Заказ не найден. Обратитесь к начальнику смены", str(order_id))
            return

        for position in order:
            (
                order_id,
                pos_id,
                row,
                rack,
                tier,
                cell,
                item,
                quantity,
                status,
            ) = position
            command = self.generate_command_text(row, rack, tier, cell, item, quantity)
            expected = {
                "row": row,
                "rack": rack,
                "tier": tier,
                "cell": cell,
                "item": item,
                "quantity": quantity,
            }
            logger.info(f"Озвучивание команды для позиции {pos_id} заказа {order_id}...")

            success, command_audio_path = self.synthesize_speech(
                f"Переходим к отбору позиции {digits_to_words(str(pos_id))} "
                f"заказа {digits_to_words(str(order_id))}",
                str(order_id),
                str(pos_id)
            )

            success, cmd_audio_path = self.synthesize_speech(
                command,
                str(order_id),
                str(pos_id)
            )
            command_audio_path = cmd_audio_path if cmd_audio_path else command_audio_path

            order_data = self.db.get_all_orders(order_id)
            plot_warehouse(order_data, position, row_filter=int(row), highlight_mode="init")

            processed = False
            logger.info("Ожидание первичного ответа...")

            while not processed:
                audio_result = self.record_audio(str(order_id), str(pos_id))
                if not audio_result:
                    self.synthesize_speech("Ошибка записи. Попробуйте снова", str(order_id), str(pos_id))
                    continue

                response_audio_path, audio_data = audio_result
                temp_file = "temp_response.wav"
                write(temp_file, SAMPLE_RATE_ASR, audio_data)

                response = self.recognize_speech(temp_file)
                os.remove(temp_file)
                if not response:
                    self.synthesize_speech("Ответ не распознан. Пожалуйста, повторите", str(order_id), str(pos_id))
                    continue

                response_data = self.extract_response_data(response)
                response_type = response_data["response_type"]

                if response_type == "Повтори команду":
                    self.synthesize_speech(command, str(order_id), str(pos_id))
                    plot_warehouse(order_data, position, row_filter=int(row), highlight_mode="init")
                    continue
                elif response_type == "Принято":
                    logger.info(f"Задержка на {PICKING_DELAY} секунд для выполнения отбора...")
                    self.synthesize_speech(
                        f"Выполняйте отбор в течение следующих "
                        f"{digits_to_words(str(PICKING_DELAY))} секунд",
                        str(order_id),
                        str(pos_id)
                    )
                    time.sleep(PICKING_DELAY)
                else:
                    self.synthesize_speech(
                        "Неизвестный тип ответа. Ожидается 'Принято' или 'Повтори команду'. "
                        "Пожалуйста, повторите",
                        str(order_id),
                        str(pos_id)
                    )
                    continue

                self.synthesize_speech("Подтвердите отбор текущей позиции", str(order_id), str(pos_id))
                attempts = 0
                max_attempts = 5
                logger.info("Ожидание последующего ответа...")

                while not processed and attempts < max_attempts:
                    attempts += 1
                    logger.info(
                        f"Попытка {attempts} для обработки ответа по отбору позиции "
                        f"{pos_id} заказа {order_id}."
                    )

                    audio_result = self.record_audio(str(order_id), str(pos_id))
                    if not audio_result:
                        self.synthesize_speech("Ошибка записи. Попробуйте снова", str(order_id), str(pos_id))
                        continue

                    resp_audio_path, audio_data = audio_result
                    temp_file = "temp_response.wav"
                    write(temp_file, SAMPLE_RATE_ASR, audio_data)

                    response = self.recognize_speech(temp_file)
                    os.remove(temp_file)
                    if not response:
                        self.synthesize_speech("Ответ не распознан. Пожалуйста, повторите", str(order_id), str(pos_id))
                        continue

                    response_data = self.extract_response_data(response)
                    response_type = response_data["response_type"]
                    response_entities = response_data["entities"]

                    self.db.save_audio_log(
                        order_id=str(order_id),
                        position_id=str(pos_id),
                        command_audio_path=command_audio_path,
                        response_audio_path=resp_audio_path,
                        command_text=command,
                        response_text=response,
                    )

                    if response_type == "Повтори команду":
                        self.synthesize_speech(command, str(order_id), str(pos_id))
                        order_data = self.db.get_all_orders(order_id)
                        plot_warehouse(order_data, position, row_filter=int(row), highlight_mode="init")
                        continue

                    if response_type == "Готово":
                        if not response_entities:
                            continue
                        if self.validate_response(response_entities, expected):
                            self.db.update_status(order_id, pos_id, "completed")
                            self.synthesize_speech(
                                f"Обновлено: позиция {digits_to_words(str(pos_id))} заказа "
                                f"{digits_to_words(str(order_id))} обработана. "
                                f"Перехожу к следующей позиции",
                                str(order_id),
                                str(pos_id)
                            )
                            order_data = self.db.get_all_orders(order_id)
                            print_order_table(order_data)
                            plot_warehouse(order_data, position, row_filter=int(row), highlight_mode="status")
                            processed = True
                        else:
                            self.synthesize_speech(
                                "Данные не совпадают. Повторите ответ или скажите 'Повтори команду'",
                                str(order_id),
                                str(pos_id)
                            )
                        continue

                    if response_type == "Товар не найден":
                        self.db.update_status(order_id, pos_id, "not_found")
                        self.synthesize_speech(
                            f"Обновлено: товар {digits_to_words(str(item))} не найден. "
                            f"Перехожу к следующей позиции",
                            str(order_id),
                            str(pos_id)
                        )
                        order_data = self.db.get_all_orders(order_id)
                        print_order_table(order_data)
                        plot_warehouse(order_data, position, row_filter=int(row), highlight_mode="status")
                        processed = True
                        continue

                    if response_type == "Не хватает единиц товара":
                        self.synthesize_speech("Назовите доступное количество", str(order_id), str(pos_id))
                        qty = None
                        for attempt in range(2):
                            audio_result = self.record_audio(str(order_id), str(pos_id))
                            if not audio_result:
                                continue

                            qty_audio_path, audio_data = audio_result
                            temp_file = "temp_response.wav"
                            write(temp_file, SAMPLE_RATE_ASR, audio_data)
                            response = self.recognize_speech(temp_file)
                            os.remove(temp_file)

                            numbers = re.findall(r"\d+", normalize_number(response.lower()))
                            if numbers:
                                qty = numbers[0]
                                self.db.save_audio_log(
                                    order_id=str(order_id),
                                    position_id=str(pos_id),
                                    command_audio_path="",
                                    response_audio_path=qty_audio_path,
                                    command_text="Назовите доступное количество",
                                    response_text=response,
                                )
                                break
                            if attempt < 1:
                                self.synthesize_speech(
                                    "Количество не распознано. Пожалуйста, повторите",
                                    str(order_id),
                                    str(pos_id)
                                )
                        if qty:
                            self.db.update_status(order_id, pos_id, f"partial_{qty}")
                            self.synthesize_speech(
                                f"Обновлено: товар {digits_to_words(str(item))}, "
                                f"{digits_to_words(str(qty))} единиц. "
                                f"Перехожу к следующей позиции",
                                str(order_id),
                                str(pos_id)
                            )
                            order_data = self.db.get_all_orders(order_id)
                            print_order_table(order_data)
                            plot_warehouse(order_data, position, row_filter=int(row), highlight_mode="status")
                        else:
                            self.db.update_status(order_id, pos_id, "error")
                            self.synthesize_speech(
                                "Не удалось распознать количество. Обратитесь к начальнику смены",
                                str(order_id),
                                str(pos_id)
                            )
                            order_data = self.db.get_all_orders(order_id)
                            print_order_table(order_data)
                            plot_warehouse(order_data, position, row_filter=int(row), highlight_mode="status")
                        processed = True
                        continue

                    if response_type == "Товар бракован":
                        self.synthesize_speech(
                            f"Поместите товар в зону брака, {BRAKED_ZONE}. "
                            f"Затем скажите 'Готово' для подтверждения",
                            str(order_id),
                            str(pos_id)
                        )
                        logger.info(
                            f"Задержка на {PICKING_DELAY} секунд для транспортировки в зону брака..."
                        )
                        time.sleep(PICKING_DELAY)
                        confirmed = False
                        for attempt in range(2):
                            audio_result = self.record_audio(str(order_id), str(pos_id))
                            if not audio_result:
                                continue

                            confirm_audio_path, audio_data = audio_result
                            temp_file = "temp_response.wav"
                            write(temp_file, SAMPLE_RATE_ASR, audio_data)
                            response = self.recognize_speech(temp_file)
                            os.remove(temp_file)

                            if self.verification.lower() in response.lower().strip():
                                confirmed = True
                                self.db.save_audio_log(
                                    order_id=str(order_id),
                                    position_id=str(pos_id),
                                    command_audio_path="",
                                    response_audio_path=confirm_audio_path,
                                    command_text="Подтверждение размещения в зоне брака",
                                    response_text=response,
                                )
                                break
                            if attempt < 1:
                                self.synthesize_speech(
                                    "Подтверждение не получено. Пожалуйста, повторите",
                                    str(order_id),
                                    str(pos_id)
                                )
                        if confirmed:
                            self.db.update_status(order_id, pos_id, "defective")
                            self.synthesize_speech(
                                f"Обновлено: товар {digits_to_words(str(item))} бракован. "
                                f"Перехожу к следующей позиции",
                                str(order_id),
                                str(pos_id)
                            )
                            order_data = self.db.get_all_orders(order_id)
                            print_order_table(order_data)
                            plot_warehouse(order_data, position, row_filter=int(row), highlight_mode="status")
                        else:
                            self.db.update_status(order_id, pos_id, "error")
                            self.synthesize_speech(
                                "Подтверждение не получено. Обратитесь к начальнику смены",
                                str(order_id),
                                str(pos_id)
                            )
                            order_data = self.db.get_all_orders(order_id)
                            print_order_table(order_data)
                            plot_warehouse(order_data, position, row_filter=int(row), highlight_mode="status")
                        processed = True
                        continue

                    self.synthesize_speech(
                        "Неизвестный тип ответа. Ожидается 'Готово', 'Товар не найден', "
                        "'Не хватает единиц товара', 'Товар бракован' или 'Повтори команду'. "
                        "Пожалуйста, повторите",
                        str(order_id),
                        str(pos_id)
                    )

                if not processed and attempts >= max_attempts:
                    self.db.update_status(order_id, pos_id, "error")
                    self.synthesize_speech(
                        "Слишком много попыток. Обратитесь к начальнику смены",
                        str(order_id),
                        str(pos_id)
                    )
                    order_data = self.db.get_all_orders(order_id)
                    print_order_table(order_data)
                    plot_warehouse(order_data, position, row_filter=int(row), highlight_mode="status")
                    processed = True

        self.synthesize_speech(
            f"Заказ {digits_to_words(str(order_id))} собран. Перейдите в зону упаковки",
            str(order_id)
        )
        order_data = self.db.get_all_orders(order_id)
        print_order_table(order_data)
        logger.info(f"Заказ {order_id} завершен.")

    def run(self, order_id: Union[str, int]) -> None:
        try:
            order_data = self.db.get_all_orders(order_id)
            if not order_data:
                logger.error(f"Заказ {order_id} не найден в базе данных (!)")
                return
            print_order_table(order_data)
            self.process_order(order_id)
        except Exception as e:
            logger.error(f"Ошибка выполнения: {e} (!)")
        finally:
            self.db.close()


def init_test_db() -> None:
    db = None
    try:
        db = OrderDatabase(DB_PATH)
        db.cursor.execute("DELETE FROM orders")
        db.cursor.execute("DELETE FROM audio_logs")
        test_order = [
            (1, 1, 1, 1, 1, 12, "1234", 3),
            (1, 2, 2, 2, 2, 15, "5678", 2),
            (1, 3, 10, 1, 1, 3, "9012", 7),
            (1, 4, 15, 2, 1, 8, "9123", 5),
            (1, 5, 5, 1, 2, 7, "3456", 4),
        ]

        db.cursor.executemany(
            """
            INSERT INTO orders (
                order_id, position, row, rack, tier, cell, item, quantity, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')
            """,
            test_order,
        )
        db.conn.commit()
        logger.info("Тестовая база данных успешно инициализирована.")
    except Exception as e:
        logger.error(f"Ошибка инициализации тестовой базы данных: {e} (!)")
    finally:
        if db:
            db.close()


if __name__ == "__main__":
    init_test_db()
    system = VoiceOrderSystem(DB_PATH)
    system.run(order_id=1)
