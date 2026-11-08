##
## EPITECH PROJECT, 2026
## my_torch
## File description:
## Makefile
##

ANALYZER_NAME = my_torch_analyzer
GENERATOR_NAME = my_torch_generator
PYTHON ?= python3

all: analyzer generator

install-deps:
	$(PYTHON) -m pip install -r requierements.txt

analyzer: install-deps $(ANALYZER_NAME)

generator: install-deps $(GENERATOR_NAME)

$(ANALYZER_NAME): src/torch_analyzer/main.py
	cp $< $@
	chmod a+x $@

$(GENERATOR_NAME): src/torch_generator/main.py
	cp $< $@
	chmod a+x $@

clean:
	$(RM) $(ANALYZER_NAME) $(GENERATOR_NAME)

fclean: clean

re: fclean
	$(MAKE) all

.PHONY: all install-deps analyzer generator clean fclean re
