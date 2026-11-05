##
## EPITECH PROJECT, 2026
## my_torch
## File description:
## Makefile
##

ANALYSER_NAME = my_torch_analyser
GENERATOR_NAME = my_torch_generator

all: analyser generator

analyser: $(ANALYSER_NAME)

generator: $(GENERATOR_NAME)

$(ANALYSER_NAME): src/torch_analyser/main.py
	cp $< $@
	chmod a+x $@

$(GENERATOR_NAME): src/torch_generator/main.py
	cp $< $@
	chmod a+x $@

clean:
	$(RM) $(ANALYSER_NAME) $(GENERATOR_NAME)

fclean: clean

re: fclean
	$(MAKE) all

.PHONY: all analyser generator clean fclean re
