import logging

class Logger:
    def __init__(self, name='SmartSupportBot'):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(name)

    def log_event(self, message):
        self.logger.info(message)

    def log_error(self, message):
        self.logger.error(message)

    def log_handoff(self, from_agent, to_agent, query):
        self.logger.info(f'Handoff from {from_agent} to {to_agent} for query: {query}')